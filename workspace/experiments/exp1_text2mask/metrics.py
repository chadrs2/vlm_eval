import csv
import numpy as np
import os

# ------------------------------------------------------------------------------------------
# ----------------------------------------- Helper -----------------------------------------
# ------------------------------------------------------------------------------------------

# Default IoU thresholds used for AP, matching the standard COCO mask-AP
# convention (AP averaged over IoU 0.50:0.05:0.95). Callers can override this
# (e.g. pass [0.5] for a plain "AP50") via evaluate_batch(iou_thresholds=...).
DEFAULT_AP_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def _unpack_instance(instance):
    """
    Predicted-mask entries are now (mask, score) tuples (confidence-aware
    predictions). Accept plain masks too, defaulting their score to 1.0, so
    older helper modules that haven't been updated yet don't hard-crash.
    """
    if isinstance(instance, (tuple, list)) and len(instance) == 2 and not np.isscalar(instance[0]):
        mask, score = instance
        return np.asarray(mask), float(score)
    return np.asarray(instance), 1.0


def _build_semantic_masks(pred_masks, gt_masks, gt_class_ids, image_shape, class_ids, void_class_ids=None):
    """
    Convert instance masks into per-class binary semantic masks.
    Used for the pixel-level IoU / F1 metrics (unaffected by confidence
    scores -- every predicted instance is still OR'd together here).
    """

    H, W = image_shape

    build_ids = set(class_ids)
    if void_class_ids is not None:
        build_ids.update(void_class_ids)

    pred_semantic = {
        class_id: np.zeros((H, W), dtype=bool)
        for class_id in set(class_ids)
    }

    gt_semantic = {
        class_id: np.zeros((H, W), dtype=bool)
        for class_id in build_ids
    }

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    for class_id, instances in pred_masks.items():

        if class_id not in pred_semantic:
            pred_semantic[class_id] = np.zeros(
                (H, W),
                dtype=bool
            )

        for instance in instances:

            mask, _score = _unpack_instance(instance)
            mask = mask.astype(bool)

            if mask.shape != (H, W):
                raise ValueError(
                    f"Prediction mask shape {mask.shape} "
                    f"does not match image shape {(H, W)}"
                )

            pred_semantic[class_id] |= mask

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------

    for mask, class_id in zip(gt_masks, gt_class_ids):

        if class_id not in gt_semantic:
            continue

        mask = mask.astype(bool)

        if mask.shape != (H, W):
            raise ValueError(
                f"GT mask shape {mask.shape} "
                f"does not match image shape {(H, W)}"
            )

        gt_semantic[class_id] |= mask

    return pred_semantic, gt_semantic


def _compute_confusion(pred_mask, gt_mask, void_mask=None):
    """
    Compute pixel-level TP, FP, FN, TN, factoring out void areas.
    """

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if void_mask is not None:
        valid_mask = ~void_mask.astype(bool)
        pred_mask = pred_mask & valid_mask
        gt_mask = gt_mask & valid_mask

    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    tn = np.logical_and(~pred_mask, ~gt_mask).sum()

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def _save_accumulator_csv(accumulator, csv_path):
    """
    Save the current accumulated TP/FP/FN/TN values to CSV.
    """

    fieldnames = [
        "class_id",
        "TP",
        "FP",
        "FN",
        "TN",
    ]

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for class_id, stats in accumulator.items():

            if not isinstance(class_id, (int, np.integer)):
                continue

            writer.writerow({
                "class_id": class_id,
                "TP": stats["TP"],
                "FP": stats["FP"],
                "FN": stats["FN"],
                "TN": stats["TN"],
            })


# ------------------------------------------------------------------------------------------
# ------------------------------------------- AP --------------------------------------------
# ------------------------------------------------------------------------------------------
#
# This is instance-level AP (not the pixel-union semantic metrics above). For
# each class, every *predicted instance* (mask, confidence score) is greedily
# matched -- highest score first -- to the best remaining, unmatched GT
# instance whose mask IoU clears the threshold. Matched predictions are TPs,
# unmatched predictions are FPs, and any GT instance nobody matched is a FN
# (baked into the recall denominator). This is repeated per IoU threshold; a
# class's AP is the precision/recall curve integrated (101-point, COCO-style)
# and then averaged over thresholds, and AP50 that appears in the printed
# summary is same 101 point computation with just threshold 0.5.
#
# Void handling: same as the pixel-level metrics above, void pixels are
# subtracted from every predicted and GT instance mask before matching.
# An instance that's entirely void after that subtraction is dropped outright 
# rather than counted as a FN/FP--it's an ignore region, not a scoreable miss.

def _mask_iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


def _compute_iou_matrix(pred_masks, gt_masks):
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            iou_matrix[i, j] = _mask_iou(p, g)
    return iou_matrix


def _apply_void_to_ap_instances(pred_instances, gt_instances, void_mask):
    """
    Subtract void pixels out of every predicted and GT instance mask before
    IoU matching, and drop any instance that ends up entirely void (the same
    "ignore region" convention COCO uses for its crowd/ignore masks): such a
    GT instance is unscoreable either way, so counting it would inflate the
    recall denominator with something no prediction could ever legitimately
    match, and a prediction that's void end-to-end isn't a real FP either --
    it never claimed anything in a region that counts.
    """

    if void_mask is None or not void_mask.any():
        return pred_instances, gt_instances

    valid = ~void_mask.astype(bool)

    filtered_preds = []
    for mask, score in pred_instances:
        m = mask.astype(bool) & valid
        if m.any():
            filtered_preds.append((m, score))

    filtered_gts = []
    for mask in gt_instances:
        m = mask.astype(bool) & valid
        if m.any():
            filtered_gts.append(m)

    return filtered_preds, filtered_gts


def _init_ap_accumulator(class_ids, iou_thresholds):
    return {
        class_id: {
            "num_gt": 0,
            "thresholds": {
                thr: {"scores": [], "matches": []}
                for thr in iou_thresholds
            },
        }
        for class_id in set(class_ids)
    }


def _accumulate_ap_for_class(pred_instances, gt_instances, class_accumulator, iou_thresholds):
    """
    pred_instances: list of (mask, score) for one image, one class.
    gt_instances: list of masks (one per GT instance) for the same image/class.
    """

    n_gt = len(gt_instances)
    class_accumulator["num_gt"] += n_gt

    if len(pred_instances) == 0:
        return

    # Highest confidence first -- greedy matching is order-dependent, and
    # COCO-style AP always resolves ties in favor of the more confident box.
    order = sorted(range(len(pred_instances)), key=lambda i: -pred_instances[i][1])
    pred_masks_sorted = [pred_instances[i][0] for i in order]
    pred_scores_sorted = [pred_instances[i][1] for i in order]

    iou_matrix = (
        _compute_iou_matrix(pred_masks_sorted, gt_instances)
        if n_gt > 0
        else np.zeros((len(pred_masks_sorted), 0))
    )

    for thr in iou_thresholds:
        matched_gt = set()
        bucket = class_accumulator["thresholds"][thr]

        for i, score in enumerate(pred_scores_sorted):
            is_tp = 0

            if n_gt > 0:
                best_j, best_iou = -1, thr
                for j in range(n_gt):
                    if j in matched_gt:
                        continue
                    iou = iou_matrix[i, j]
                    if iou >= best_iou:
                        best_iou = iou
                        best_j = j
                if best_j >= 0:
                    matched_gt.add(best_j)
                    is_tp = 1

            bucket["scores"].append(score)
            bucket["matches"].append(is_tp)


def _compute_ap_from_scores(scores, matches, num_gt):
    """
    101-point interpolated PR-curve AP (matches the COCO mask-AP convention).
    """

    if num_gt == 0:
        return np.nan
    if len(scores) == 0:
        return 0.0

    scores = np.asarray(scores, dtype=np.float64)
    matches = np.asarray(matches, dtype=np.float64)

    order = np.argsort(-scores)
    matches = matches[order]

    tp_cum = np.cumsum(matches)
    fp_cum = np.cumsum(1.0 - matches)

    recalls = tp_cum / float(num_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Pad with the boundary points and enforce a monotonically
    # non-increasing precision envelope (standard PASCAL/COCO smoothing).
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(recall_thresholds)
    for k, rt in enumerate(recall_thresholds):
        idx = np.searchsorted(recalls, rt, side="left")
        interpolated[k] = precisions[idx] if idx < len(precisions) else 0.0

    return float(interpolated.mean())


def _finalize_ap(ap_data, category_name_dict, iou_thresholds):
    """
    Turn accumulated (scores, matches, num_gt) per class/threshold into:
      - per_class_ap[class_id]:  AP averaged over iou_thresholds
      - per_class_ap50[class_id], per_class_ap75[class_id] (if present)
      - mAP, mAP50, mAP75 averaged over classes with num_gt > 0
    Negative class ids (prompts with no GT association) are skipped, same
    convention as the pixel-level metrics.
    """

    per_class_ap, per_class_ap50, per_class_ap75 = {}, {}, {}
    all_ap, all_ap50, all_ap75 = [], [], []

    for class_id, class_name in category_name_dict.items():

        if class_id < 0 or class_id not in ap_data:
            per_class_ap[class_id] = np.nan
            per_class_ap50[class_id] = np.nan
            per_class_ap75[class_id] = np.nan
            continue

        entry = ap_data[class_id]
        num_gt = entry["num_gt"]

        thr_aps = {}
        for thr, bucket in entry["thresholds"].items():
            thr_aps[thr] = _compute_ap_from_scores(bucket["scores"], bucket["matches"], num_gt)

        valid_aps = [v for v in thr_aps.values() if not np.isnan(v)]
        class_ap = float(np.mean(valid_aps)) if valid_aps else np.nan

        class_ap50 = thr_aps.get(0.5, np.nan)
        class_ap75 = thr_aps.get(0.75, np.nan)

        per_class_ap[class_id] = class_ap
        per_class_ap50[class_id] = class_ap50
        per_class_ap75[class_id] = class_ap75

        if num_gt > 0:
            if not np.isnan(class_ap):
                all_ap.append(class_ap)
            if not np.isnan(class_ap50):
                all_ap50.append(class_ap50)
            if not np.isnan(class_ap75):
                all_ap75.append(class_ap75)

    summary = {
        "AP": float(np.mean(all_ap)) if all_ap else np.nan,
        "AP50": float(np.mean(all_ap50)) if all_ap50 else np.nan,
        "AP75": float(np.mean(all_ap75)) if all_ap75 else np.nan,
    }

    return per_class_ap, per_class_ap50, per_class_ap75, summary


# ------------------------------------------------------------------------------------------
# ---------------------------------------- Process -----------------------------------------
# ------------------------------------------------------------------------------------------

def evaluate_batch(
    batch_masks, batch_gt_masks, batch_class_ids, batch_images, class_ids,
    void_class_ids=None, accumulator=None, runtime=None, csv_path=None,
    confusion_pairs=None, iou_thresholds=None, instance_class_ids=None,
):
    """
    Evaluate one batch of predictions with void subtraction and inter-class confusion.

    batch_masks: list (per image) of {class_id: [(mask, score), ...]}. Plain
    [mask, ...] lists (score-less) are still accepted for backward
    compatibility -- they're treated as score=1.0 for every instance, which
    makes AP degenerate to "does at least one instance match" rather than a
    real ranked PR curve, so a real per-instance confidence score is what
    you want from the model helper for a proper AP number.
    """
    if confusion_pairs is None:
        confusion_pairs = []

    if iou_thresholds is None:
        iou_thresholds = DEFAULT_AP_IOU_THRESHOLDS

    if accumulator is None:
        accumulator = {
            class_id: {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0,
            }
            for class_id in set(class_ids)
        }

        accumulator["_runtime"] = {
            "total": 0.0,
            "num_images": 0,
            "num_batches": 0,
        }

        accumulator["_pairwise_confusion"] = {
            pair: np.zeros((3, 3), dtype=np.int64)
            for pair in confusion_pairs
        }

        accumulator["_ap_iou_thresholds"] = tuple(iou_thresholds)
        
        # Only initialize AP accumulators for target instance classes (or all if none specified)
        ap_eval_ids = set(class_ids) if instance_class_ids is None else set(class_ids).intersection(instance_class_ids)
        accumulator["_ap_data"] = _init_ap_accumulator(ap_eval_ids, iou_thresholds)

    # ---------------------------------------------------------
    # Accumulate runtime
    # ---------------------------------------------------------

    if runtime is not None:
        accumulator["_runtime"]["total"] += runtime

    accumulator["_runtime"]["num_images"] += len(batch_images)
    accumulator["_runtime"]["num_batches"] += 1

    # ---------------------------------------------------------
    # Evaluate every image in batch
    # ---------------------------------------------------------

    for pred_masks, gt_masks, gt_class_ids, image in zip(
        batch_masks,
        batch_gt_masks,
        batch_class_ids,
        batch_images,
    ):

        H, W = image.shape[:2]

        pred_semantic, gt_semantic = _build_semantic_masks(
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            gt_class_ids=gt_class_ids,
            image_shape=(H, W),
            class_ids=class_ids,
            void_class_ids=void_class_ids
        )

        # Build composite void mask
        void_mask = np.zeros((H, W), dtype=bool)
        if void_class_ids is not None:
            for v_id in void_class_ids:
                if v_id in gt_semantic:
                    void_mask |= gt_semantic[v_id]

        for class_id in set(class_ids):

            pred_mask = pred_semantic.get(
                class_id,
                np.zeros((H, W), dtype=bool)
            )

            gt_mask = gt_semantic.get(
                class_id,
                np.zeros((H, W), dtype=bool)
            )

            stats = _compute_confusion(
                pred_mask,
                gt_mask,
                void_mask=void_mask
            )

            accumulator[class_id]["TP"] += stats["TP"]
            accumulator[class_id]["FP"] += stats["FP"]
            accumulator[class_id]["FN"] += stats["FN"]
            accumulator[class_id]["TN"] += stats["TN"]

            # -----------------------------------------------------
            # AP: instance-level matching (uses the raw, un-unioned
            # predicted/GT instances, not the pixel-union masks above)
            # -----------------------------------------------------
            
            if instance_class_ids is None or class_id in instance_class_ids:
                pred_instances = [
                    _unpack_instance(inst) for inst in pred_masks.get(class_id, [])
                ]
                gt_instances = [
                    gm for gm, gc in zip(gt_masks, gt_class_ids) if gc == class_id
                ]

                # void class handling
                if void_mask is not None and void_mask.any():
                    valid = ~void_mask.astype(bool)
                
                    filtered_preds = []
                    for mask, score in pred_instances:
                        m = mask.astype(bool) & valid
                        if m.any():
                            filtered_preds.append((m, score))
                
                    filtered_gts = []
                    for mask in gt_instances:
                        m = mask.astype(bool) & valid
                        if m.any():
                            filtered_gts.append(m)

                    pred_instances = filtered_preds
                    gt_instances = filtered_gts

                _accumulate_ap_for_class(
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                    class_accumulator=accumulator["_ap_data"][class_id],
                    iou_thresholds=accumulator["_ap_iou_thresholds"],
                )

        # ---------------------------------------------------------
        # Compute Inter-Class Confusion (GT vs Pred Overlap)
        # ---------------------------------------------------------
        for pair in confusion_pairs:
            id_a, id_b = pair
            gt_a = gt_semantic.get(id_a, np.zeros((H, W), dtype=bool))
            gt_b = gt_semantic.get(id_b, np.zeros((H, W), dtype=bool))

            pred_a = pred_semantic.get(id_a, np.zeros((H, W), dtype=bool))
            pred_b = pred_semantic.get(id_b, np.zeros((H, W), dtype=bool))

            valid_mask = ~void_mask if void_mask is not None else np.ones((H, W), dtype=bool)

            gt_a = gt_a & valid_mask
            gt_b = gt_b & valid_mask
            pred_a = pred_a & valid_mask
            pred_b = pred_b & valid_mask

            gt_neither = ~(gt_a | gt_b) & valid_mask
            pred_neither = ~(pred_a | pred_b) & valid_mask

            matrix = accumulator["_pairwise_confusion"][pair]

            # Row 0: GT A
            matrix[0, 0] += (gt_a & pred_a).sum()
            matrix[0, 1] += (gt_a & pred_b).sum()
            matrix[0, 2] += (gt_a & pred_neither).sum()

            # Row 1: GT B
            matrix[1, 0] += (gt_b & pred_a).sum()
            matrix[1, 1] += (gt_b & pred_b).sum()
            matrix[1, 2] += (gt_b & pred_neither).sum()

            # Row 2: GT Neither
            matrix[2, 0] += (gt_neither & pred_a).sum()
            matrix[2, 1] += (gt_neither & pred_b).sum()
            matrix[2, 2] += (gt_neither & pred_neither).sum()

    if csv_path is not None:
        _save_accumulator_csv(
            accumulator=accumulator,
            csv_path=csv_path,
        )

    return accumulator


def finalize_evaluation(accumulator, category_name_dict, csv_path=None):
    """
    Compute final dataset-level metrics from an accumulator and save confusion matrix.
    """

    per_class = {}

    ious = []
    f1_scores = []

    iou_thresholds = accumulator.get("_ap_iou_thresholds", DEFAULT_AP_IOU_THRESHOLDS)
    ap_data = accumulator.get("_ap_data", {})

    per_class_ap, per_class_ap50, per_class_ap75, ap_summary = _finalize_ap(
        ap_data, category_name_dict, iou_thresholds
    )

    # ---------------------------------------------------------
    # Per-class metrics
    # ---------------------------------------------------------

    for class_id, class_name in category_name_dict.items():

        # Skip evaluation entirely for negative class ids
        if class_id < 0:
            per_class[class_id] = {
                "class_name": class_name,
                "TP": np.nan,
                "FP": np.nan,
                "FN": np.nan,
                "TN": np.nan,
                "IoU": np.nan,
                "F1-Score": np.nan,
                "AP": np.nan,
                "AP50": np.nan,
                "AP75": np.nan,
            }
            continue

        stats = accumulator.get(
            class_id,
            {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0,
            }
        )

        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]
        tn = stats["TN"]

        # IoU
        union = tp + fp + fn

        if union > 0:
            iou = tp / union
            ious.append(iou)
        else:
            iou = np.nan

        # F1 / F-measure
        denominator = (
            2 * tp +
            fp +
            fn
        )

        if denominator > 0:
            f1_score = (2 * tp) / denominator
            f1_scores.append(f1_score)
        else:
            f1_score = np.nan

        per_class[class_id] = {
            "class_name": class_name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "IoU": iou,
            "F1-Score": f1_score,
            "AP": per_class_ap.get(class_id, np.nan),
            "AP50": per_class_ap50.get(class_id, np.nan),
            "AP75": per_class_ap75.get(class_id, np.nan),
        }

    # ---------------------------------------------------------
    # Mean metrics
    # ---------------------------------------------------------

    miou = (
        np.nanmean(ious)
        if len(ious) > 0
        else np.nan
    )

    mf1_score = (
        np.nanmean(f1_scores)
        if len(f1_scores) > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # Runtime
    # ---------------------------------------------------------

    runtime_info = accumulator["_runtime"]

    if runtime_info["num_images"] > 0:
        average_runtime = (
            runtime_info["total"]
            / runtime_info["num_images"]
        )
    else:
        average_runtime = np.nan

    results = {
        "per_class": per_class,
        "mIoU": miou,
        "Mean_F1": mf1_score,
        "AP": ap_summary["AP"],
        "AP50": ap_summary["AP50"],
        "AP75": ap_summary["AP75"],
        "ap_iou_thresholds": tuple(iou_thresholds),
        "average_runtime": average_runtime,
        "total_runtime": runtime_info["total"],
        "num_images": runtime_info["num_images"],
    }

    if csv_path is not None:

        fieldnames = [
            "class_id",
            "class_name",
            "TP",
            "FP",
            "FN",
            "TN",
            "IoU",
            "F1-Score",
            "AP",
            "AP50",
            "AP75",
            "mIoU",
            "Mean_F1",
            "mAP",
            "mAP50",
            "mAP75",
            "average_runtime",
            "total_runtime",
            "num_images",
        ]

        with open(csv_path, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for class_id, stats in per_class.items():

                writer.writerow({
                    "class_id": class_id,
                    "class_name": stats["class_name"],
                    "TP": stats["TP"],
                    "FP": stats["FP"],
                    "FN": stats["FN"],
                    "TN": stats["TN"],
                    "IoU": stats["IoU"],
                    "F1-Score": stats["F1-Score"],
                    "AP": stats["AP"],
                    "AP50": stats["AP50"],
                    "AP75": stats["AP75"],
                    "mIoU": miou,
                    "Mean_F1": mf1_score,
                    "mAP": results["AP"],
                    "mAP50": results["AP50"],
                    "mAP75": results["AP75"],
                    "average_runtime": average_runtime,
                    "total_runtime": runtime_info["total"],
                    "num_images": runtime_info["num_images"],
                })

        # ---------------------------------------------------------
        # Write Pairwise 3x3 Matrices
        # ---------------------------------------------------------
        if "_pairwise_confusion" in accumulator and accumulator["_pairwise_confusion"]:
            pairwise_csv_path = csv_path.replace(".csv", "_3x3_pairwise_confusion.csv")
            with open(pairwise_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                for pair, matrix in accumulator["_pairwise_confusion"].items():
                    name_a = category_name_dict.get(pair[0], str(pair[0]))
                    name_b = category_name_dict.get(pair[1], str(pair[1]))

                    writer.writerow([f"Pair: {name_a} vs {name_b}"])
                    writer.writerow(["", f"Pred: {name_a}", f"Pred: {name_b}", "Pred: Neither"])
                    writer.writerow([f"GT: {name_a}", matrix[0, 0], matrix[0, 1], matrix[0, 2]])
                    writer.writerow([f"GT: {name_b}", matrix[1, 0], matrix[1, 1], matrix[1, 2]])
                    writer.writerow(["GT: Neither", matrix[2, 0], matrix[2, 1], matrix[2, 2]])
                    writer.writerow([])

            results["pairwise_matrix_path"] = pairwise_csv_path

    return results


# ------------------------------------------------------------------------------------------
# ----------------------------------------- Display ----------------------------------------
# ------------------------------------------------------------------------------------------

def print_evaluation_results(results, model_name, dataset_name):
    """
    Pretty-print evaluation results, sorted alphabetically and grouped by evaluation status.
    """

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {model_name} ({dataset_name})")
    print("=" * 80)

    print(
        f"mIoU:             {results['mIoU']:.4f}"
    )

    print(
        f"Mean_F1-Score:    {results['Mean_F1']:.4f}"
    )

    print(
        f"AP (0.50:0.95):   {results['AP']:.4f}"
    )

    print(
        f"AP50:             {results['AP50']:.4f}"
    )

    print(
        f"AP75:             {results['AP75']:.4f}"
    )

    print(
        f"Average runtime:  "
        f"{results['average_runtime']:.4f} sec/image"
    )

    print(
        f"Total runtime:    "
        f"{results['total_runtime']:.4f} sec"
    )

    print(
        f"Images evaluated: "
        f"{results['num_images']}"
    )

    print("\nPer-class metrics:")
    print("-" * 92)

    header = (
        f"{'Class':<22}"
        f"{'TP':>10}"
        f"{'FP':>10}"
        f"{'FN':>10}"
        f"{'TN':>10}"
        f"{'IoU':>8}"
        f"{'F1':>8}"
        f"{'AP':>8}"
        f"{'AP50':>8}"
    )

    print(header)
    print("-" * 92)

    # Separate classes into evaluated (ID >= 0) and unevaluated (ID < 0)
    evaluated_classes = []
    unevaluated_classes = []

    for class_id, stats in results["per_class"].items():
        if class_id >= 0:
            evaluated_classes.append(stats)
        else:
            unevaluated_classes.append(stats)

    # Sort evaluated classes alphabetically
    evaluated_classes.sort(key=lambda x: x["class_name"].lower())

    for stats in evaluated_classes:

        print(
            f"{stats['class_name']:<22}"
            f"{stats['TP']:>10}"
            f"{stats['FP']:>10}"
            f"{stats['FN']:>10}"
            f"{stats['TN']:>10}"
            f"{stats['IoU']:>8.3f}"
            f"{stats['F1-Score']:>8.3f}"
            f"{stats['AP']:>8.3f}"
            f"{stats['AP50']:>8.3f}"
        )

    # Output unevaluated classes if they exist
    if unevaluated_classes:
        print("\n" + "-" * 92)
        print(f"{'Unevaluated Classes (Not in Ground Truth)':<92}")
        print("-" * 92)
        for stats in unevaluated_classes:
            print(
                f"{stats['class_name']:<22}"
                f"{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>8}{'-':>8}{'-':>8}{'-':>8}"
            )

    print("=" * 92)

    if "pairwise_matrix_path" in results:
        print(f"\n--> Inter-class confusion matrix saved to:")
        print(f"    {results['pairwise_matrix_path']}")