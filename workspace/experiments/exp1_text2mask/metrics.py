import csv
import numpy as np

# from sklearn.metrics import average_precision_score


def _build_semantic_masks(
    pred_masks,
    gt_masks,
    gt_class_ids,
    image_shape,
    class_ids,
):
    """
    Convert instance masks into per-class binary semantic masks.

    Args:
        pred_masks:
            Dict:
                {
                    class_id: [mask1, mask2, ...]
                }

        gt_masks:
            List of GT instance masks.

        image_shape:
            (H, W)

        class_ids:
            GT class ID corresponding to each mask in gt_masks.

    Returns:
        pred_semantic:
            Dict[class_id] -> binary HxW mask

        gt_semantic:
            Dict[class_id] -> binary HxW mask
    """

    H, W = image_shape

    pred_semantic = {
        class_id: np.zeros((H, W), dtype=bool)
        for class_id in class_ids
    }

    gt_semantic = {
        class_id: np.zeros((H, W), dtype=bool)
        for class_id in class_ids
    }

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    for class_id, masks in pred_masks.items():

        if class_id not in pred_semantic:
            pred_semantic[class_id] = np.zeros(
                (H, W),
                dtype=bool
            )

        for mask in masks:

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
            gt_semantic[class_id] = np.zeros(
                (H, W),
                dtype=bool
            )

        mask = mask.astype(bool)

        if mask.shape != (H, W):
            raise ValueError(
                f"GT mask shape {mask.shape} "
                f"does not match image shape {(H, W)}"
            )

        gt_semantic[class_id] |= mask

    return pred_semantic, gt_semantic


def _compute_confusion(
    pred_mask,
    gt_mask,
):
    """
    Compute pixel-level TP, FP, FN, TN.
    """

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

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

def _save_accumulator_csv(
    accumulator,
    csv_path,
):
    """
    Save the current accumulated TP/FP/FN/TN values to CSV.

    This is primarily useful for monitoring evaluation progress
    while batches are being processed.
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

            # Skip metadata entries such as "_runtime"
            if not isinstance(class_id, (int, np.integer)):
                continue

            writer.writerow({
                "class_id": class_id,
                "TP": stats["TP"],
                "FP": stats["FP"],
                "FN": stats["FN"],
                "TN": stats["TN"],
            })

def evaluate_batch(
    batch_masks,
    batch_gt_masks,
    batch_class_ids,
    batch_images,
    class_ids,
    accumulator=None,
    runtime=None,
    csv_path=None,
):
    """
    Evaluate one batch of predictions.

    This function accumulates pixel-level TP/FP/FN/TN across
    images so that metrics can be computed over the entire
    dataset.

    Args:
        batch_masks:
            List of prediction dictionaries, one per image.

            Example:
                [
                    {
                        2: [mask1, mask2],
                        7: [mask3],
                    },
                    ...
                ]

        batch_gt_masks:
            List of GT instance-mask lists, one per image.

        batch_class_ids:
            List of GT class-ID lists, one per image.

        batch_images:
            List of images.

        class_ids:
            All class IDs being evaluated.

        accumulator:
            Existing accumulator returned by this function.
            Pass None for the first batch.

        runtime:
            Runtime in seconds for generating this batch's
            predictions.

    Returns:
        accumulator
    """

    if accumulator is None:
        accumulator = {
            class_id: {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0,
            }
            for class_id in class_ids
        }

        accumulator["_runtime"] = {
            "total": 0.0,
            "num_images": 0,
            "num_batches": 0,
        }

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
        )

        for class_id in class_ids:

            # Make sure both masks exist
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
            )

            accumulator[class_id]["TP"] += stats["TP"]
            accumulator[class_id]["FP"] += stats["FP"]
            accumulator[class_id]["FN"] += stats["FN"]
            accumulator[class_id]["TN"] += stats["TN"]
    
    if csv_path is not None:
        _save_accumulator_csv(
            accumulator=accumulator,
            csv_path=csv_path,
        )

    return accumulator


def finalize_evaluation(
    accumulator,
    category_name_dict,
    csv_path=None,
):
    """
    Compute final dataset-level metrics from an accumulator.

    Returns:
        results dictionary containing:

            per_class:
                {
                    class_id: {
                        "class_name": ...,
                        "TP": ...,
                        "FP": ...,
                        "FN": ...,
                        "TN": ...,
                        "IoU": ...,
                    }
                }

            mIoU
            F-mIoU
            AP
            average_runtime
    """

    per_class = {}

    ious = []
    f1_scores = []

    # ---------------------------------------------------------
    # Per-class metrics
    # ---------------------------------------------------------

    for class_id, class_name in category_name_dict.items():

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
        "AP": np.nan,  # Requires confidence scores
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
            "mIoU",
            "Mean_F1",
            "AP",
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
                    "mIoU": miou,
                    "Mean_F1": mf1_score,
                    "AP": results["AP"],
                    "average_runtime": average_runtime,
                    "total_runtime": runtime_info["total"],
                    "num_images": runtime_info["num_images"],
                })
    
    return results


def print_evaluation_results(results):
    """
    Pretty-print evaluation results.
    """

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"mIoU:             {results['mIoU']:.4f}"
    )

    print(
        f"Mean_F1-Score:           {results['Mean_F1']:.4f}"
    )

    print(
        f"AP:               {results['AP']:.4f}"
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
    print("-" * 80)

    header = (
        f"{'Class':<25}"
        f"{'TP':>12}"
        f"{'FP':>12}"
        f"{'FN':>12}"
        f"{'TN':>12}"
        f"{'IoU':>10}"
        f"{'F1-Score':>10}"
    )

    print(header)
    print("-" * 80)

    for class_id, stats in results["per_class"].items():

        print(
            f"{stats['class_name']:<25}"
            f"{stats['TP']:>12}"
            f"{stats['FP']:>12}"
            f"{stats['FN']:>12}"
            f"{stats['TN']:>12}"
            f"{stats['IoU']:>10.4f}"
            f"{stats['F1-Score']:>10.4f}"
        )

    print("=" * 80)