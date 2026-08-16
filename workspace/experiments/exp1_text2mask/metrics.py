import csv
import numpy as np
import os

# ------------------------------------------------------------------------------------------
# ----------------------------------------- Helper -----------------------------------------
# ------------------------------------------------------------------------------------------

def _build_semantic_masks( pred_masks, gt_masks, gt_class_ids, image_shape, class_ids, void_class_ids=None):
    """
    Convert instance masks into per-class binary semantic masks.
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
# ---------------------------------------- Process -----------------------------------------
# ------------------------------------------------------------------------------------------

def evaluate_batch(batch_masks, batch_gt_masks, batch_class_ids, batch_images, class_ids, void_class_ids=None, accumulator=None, runtime=None, csv_path=None, confusion_pairs=None):
    """
    Evaluate one batch of predictions with void subtraction and inter-class confusion.
    """
    if confusion_pairs is None:
        confusion_pairs = []

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
        "AP": np.nan,  
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
            f"{stats['class_name']:<25}"
            f"{stats['TP']:>12}"
            f"{stats['FP']:>12}"
            f"{stats['FN']:>12}"
            f"{stats['TN']:>12}"
            f"{stats['IoU']:>10.4f}"
            f"{stats['F1-Score']:>10.4f}"
        )

    # Output unevaluated classes if they exist
    if unevaluated_classes:
        print("\n" + "-" * 80)
        print(f"{'Unevaluated Classes (Not in Ground Truth)':<80}")
        print("-" * 80)
        for stats in unevaluated_classes:
            print(
                f"{stats['class_name']:<25}"
                f"{'-':>12}"
                f"{'-':>12}"
                f"{'-':>12}"
                f"{'-':>12}"
                f"{'-':>10}"
                f"{'-':>10}"
            )

    print("=" * 80)
    
    if "pairwise_matrix_path" in results:
        print(f"\n--> Inter-class confusion matrix saved to:")
        print(f"    {results['pairwise_matrix_path']}")