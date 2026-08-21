import os
from argparse import ArgumentParser
import glob
import random
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from pycocotools.coco import COCO

# Import metrics logic originally held in the helper
from metrics1 import (
    evaluate_batch,
    finalize_evaluation,
    print_evaluation_results,
    DEFAULT_AP_IOU_THRESHOLDS,
)


def load_data(dataset_dir, annotation_file=None, image_subset=-1):
    """Load COCO annotations, YAML config, and return dataset and parameters with custom mappings."""

    dataset_dir = os.path.abspath(dataset_dir)

    # 1. Load YAML Configuration
    yaml_files = glob.glob(os.path.join(dataset_dir, "*.yaml")) + glob.glob(os.path.join(dataset_dir, "*.yml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml configuration file found in {dataset_dir}")

    print("Using config file:", yaml_files[0])
    with open(yaml_files[0], 'r') as f:
        config = yaml.safe_load(f) or {}

    ignore_classes = config.get("ignore_classes", []) or []
    void_classes = config.get("void_classes", []) or []
    instance_classes = config.get("instance_classes", None)
    custom_prompts = config.get("custom_prompts", None)
    confusion_pairs_yaml = config.get("confusion_pairs", [])
    custom_ground_truth_yaml = config.get("custom_ground_truth", []) or []

    # Experiment-2-only knobs (top-k retrieval config). Left as raw YAML
    # values (None if absent) -- run_experiment2.py owns defaulting/parsing
    # since run_experiment1.py (Experiment 1) has no use for them itself.
    exp2_top_k = config.get("top_k", None)
    exp2_min_class_count = config.get("min_class_count", None)
    exp2_plot_k = config.get("plot_k", None)

    # 2. Load JSON Annotations
    if annotation_file is None:
        json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json annotation file found in {dataset_dir}")
        annotation_file = json_files[0]

    print("Using annotation file:", annotation_file)
    coco = COCO(annotation_file)

    categories = coco.loadCats(coco.getCatIds())
    category_names = {cat["id"]: cat["name"] for cat in categories}
    name_to_id = {cat["name"]: cat["id"] for cat in categories}

    # 3. Process Custom Ground Truth Mapping (with Validation)
    coco_id_remap = {}

    # Generate new IDs for custom target classes that don't exist natively
    next_custom_id = max(name_to_id.values()) + 1 if name_to_id else 1

    for item in custom_ground_truth_yaml:
        if isinstance(item, dict):
            for target_class, source_string in item.items():

                # Register target class if it doesn't exist natively
                if target_class not in name_to_id:
                    name_to_id[target_class] = next_custom_id
                    category_names[next_custom_id] = target_class
                    next_custom_id += 1

                target_id = name_to_id[target_class]

                sources = [s.strip() for s in str(source_string).split(",")]
                for src in sources:
                    if src not in name_to_id:
                        # Throw an error if a mapped source class isn't real
                        raise ValueError(f"Custom ground truth mapping error: Source class '{src}' for target '{target_class}' does not exist in dataset annotations.")

                    # Store the remapping relationship
                    coco_id_remap[name_to_id[src]] = target_id

    # 4. Process Ignore and Void Classes (Strings to IDs)

    ### a. extract auto info from yaml file
    active_target_classes = set()

    # Safely extract target names from custom_prompts (which is a list)
    if isinstance(custom_prompts, list):
        for item in custom_prompts:
            if isinstance(item, dict):
                active_target_classes.update(item.keys())
            elif isinstance(item, str):
                active_target_classes.add(item)
    elif isinstance(custom_prompts, dict):
        active_target_classes.update(custom_prompts.keys())

    # Extract from custom_ground_truth_yaml
    for item in custom_ground_truth_yaml:
        if isinstance(item, dict):
            for target, source_str in item.items():
                active_target_classes.add(target)
                active_target_classes.update([s.strip() for s in str(source_str).split(",")])

    # Extract from instance_classes. Experiment 1 YAMLs normally already list
    # every instance class inside custom_prompts too, so this is a no-op for
    # them. Experiment 2 YAMLs have no custom_prompts at all -- instance_classes
    # is the *only* signal of which classes matter -- so without this, every
    # class (including the instance classes) gets swept into ignore_classes
    # below and every GT annotation is dropped.
    if instance_classes is not None:
        active_target_classes.update(instance_classes)

    # Automatically add everything else to ignore_classes
    for cls_name in name_to_id.keys():
        if cls_name not in active_target_classes and cls_name not in void_classes:
            if cls_name not in ignore_classes:
                ignore_classes.append(cls_name)

    ### b. extract manual info from yaml file
    ignore_ids = set()
    for cls_name in ignore_classes:
        if cls_name not in name_to_id:
            raise ValueError(f"Ignore class '{cls_name}' not found in dataset annotations.")
        ignore_ids.add(name_to_id[cls_name])

    void_class_ids = []
    for cls_name in void_classes:
        if cls_name not in name_to_id:
            raise ValueError(f"Void class '{cls_name}' not found in dataset annotations.")
        void_class_ids.append(name_to_id[cls_name])

    instance_class_ids = None
    if instance_classes is not None:
        instance_class_ids = []
        for cls_name in instance_classes:
            if cls_name not in name_to_id:
                raise ValueError(f"Instance class '{cls_name}' not found in dataset annotations.")
            instance_class_ids.append(name_to_id[cls_name])

    gt_class_dict = {k: v for k, v in category_names.items() if k not in ignore_ids}

    # 5. Standardize Custom Prompts Mapping
    if custom_prompts is None:
        custom_prompts = {
            name: name for class_id, name in gt_class_dict.items()
            if class_id not in void_class_ids
        }
    elif isinstance(custom_prompts, dict):
        for k, v in custom_prompts.items():
            if v is None:
                custom_prompts[k] = k
    elif isinstance(custom_prompts, list):
        merged_prompts = {}
        for item in custom_prompts:
            if isinstance(item, dict):
                for k, v in item.items():
                    merged_prompts[k] = v if v is not None else k
            elif isinstance(item, str):
                merged_prompts[item] = item
        custom_prompts = merged_prompts

    # 6. Collect image metadata
    image_ids = coco.getImgIds()
    if image_subset > 0:
        image_ids = image_ids[:image_subset]

    image_paths = []
    ann_ids_list = []

    for image_id in image_ids:
        img_meta = coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset_dir, img_meta["file_name"])
        image_paths.append(image_path)
        ann_ids_list.append(coco.getAnnIds(imgIds=[image_id]))

    return (
        image_paths, ann_ids_list, coco, category_names, ignore_ids,
        gt_class_dict, custom_prompts, void_class_ids, confusion_pairs_yaml,
        instance_class_ids, coco_id_remap, exp2_top_k, exp2_min_class_count, exp2_plot_k
    )

# ------------------------------------------------------------------------------------------
# ----------------------------------- Prompt Mapping ----------------------------------------
# ------------------------------------------------------------------------------------------

def build_prompt_mapping(custom_prompts, name_to_id):
    """
    Turn a standardized custom_prompts dict (target_class_name -> comma-
    separated prompt string, as produced by load_data's "Standardize Custom
    Prompts Mapping" step above) into:

      prompt_category_dict: {target_id: target_class_name}, one entry per
        target class. target_id is the class's real COCO category id when
        target_class_name matches an existing dataset category (via
        name_to_id); otherwise a synthetic negative id is minted for it --
        a prompt with no matching GT category (e.g. a pure distractor
        prompt meant to compete against real classes, never itself
        scoreable against GT).

      prompt_groups: {target_id: [prompt_str, ...]}, that class's
        individual (comma-split) prompt phrases, in YAML order.

    This is exactly the target-class/prompt-id bookkeeping this file's
    __main__ below does to build prompt_category_dict / prompt_class_ids /
    prompt_class_names for Experiment 1's text-to-mask prediction -- pulled
    out into its own function so other experiments (e.g. run_experiment3.py)
    can reuse the same target-class definitions instead of duplicating this
    logic or relying on a separate `instance_classes` YAML key.
    """
    prompt_category_dict = {}
    prompt_groups = {}
    neg_id_counter = -1

    for target_class, prompt_string in custom_prompts.items():
        if target_class in name_to_id:
            prompt_id = name_to_id[target_class]
        else:
            prompt_id = neg_id_counter
            neg_id_counter -= 1

        prompt_category_dict[prompt_id] = target_class
        prompt_groups[prompt_id] = [p.strip() for p in str(prompt_string).split(",") if p.strip()]

    return prompt_category_dict, prompt_groups

# ------------------------------------------------------------------------------------------
# --------------------------------------- Lazy Decode --------------------------------------
# ------------------------------------------------------------------------------------------

def load_image_and_gt(image_path, ann_ids, coco, category_names, ignore_ids, coco_id_remap=None):
    """Decode a single image + its ground-truth instance masks on demand with remapping."""
    if coco_id_remap is None:
        coco_id_remap = {}

    image = np.array(Image.open(image_path).convert("RGB"))
    annotations = coco.loadAnns(ann_ids)

    image_masks, image_class_ids, image_class_names = [], [], []

    for ann in annotations:
        category_id = int(ann["category_id"])
        if category_id in ignore_ids:
            continue

        # Remap category ID if specified in custom ground truth settings
        category_id = coco_id_remap.get(category_id, category_id)

        mask = coco.annToMask(ann)
        mask = (mask > 0).astype(np.uint8)

        image_masks.append(mask)
        image_class_ids.append(category_id)
        image_class_names.append(category_names.get(category_id, "unknown"))

    return image, image_masks, image_class_ids, image_class_names

# ------------------------------------------------------------------------------------------
# ----------------------------------------- Debug ------------------------------------------
# ------------------------------------------------------------------------------------------

def debug_visualize_random_image(image_paths, ann_ids_list, coco, category_names, ignore_ids, coco_id_remap=None, alpha=0.45):
    """Randomly visualize one image with custom-remapped instance masks."""
    if len(image_paths) == 0:
        print("No images available.")
        return

    image_idx = random.randrange(len(image_paths))
    image, image_masks, image_class_ids, image_class_names = load_image_and_gt(
        image_paths[image_idx], ann_ids_list[image_idx], coco, category_names, ignore_ids, coco_id_remap
    )

    print(f"Visualizing image index: {image_idx}")
    print(f"Image shape: {image.shape}")
    print(f"Number of masks: {len(image_masks)}")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(image)
    cmap = plt.get_cmap("tab20")

    for mask_idx, mask in enumerate(image_masks):
        mask = mask.astype(bool)
        if not np.any(mask):
            continue

        color = cmap(mask_idx % 20)[:3]
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
        colored_mask[..., :3] = color
        colored_mask[..., 3] = mask * alpha
        ax.imshow(colored_mask)

        ys, xs = np.where(mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        width, height = x_max - x_min, y_max - y_min

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        class_name = image_class_names[mask_idx]
        class_id = image_class_ids[mask_idx]
        label = f"{mask_idx}: {class_name} (id={class_id})"
        ax.text(x_min, max(0, y_min - 5), label, color="white", fontsize=10, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))

    ax.set_title(f"Image {image_idx} — {len(image_masks)} masks")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def debug_visualize_predictions(
    batch_images, batch_image_paths, batch_masks,
    prompt_class_ids, prompt_class_names, target_image_name, alpha=0.45
):
    if not target_image_name:
        return

    for i, img_path in enumerate(batch_image_paths):
        if target_image_name in img_path:
            print(f"\n--- DEBUGGER TRIGGERED FOR: {target_image_name} ---")
            img = batch_images[i]
            pred_masks_dict = batch_masks[i]

            fig, ax = plt.subplots(figsize=(14, 10))
            ax.imshow(img)
            cmap = plt.get_cmap("tab20")

            color_idx = 0
            for class_id, instances in pred_masks_dict.items():
                class_name = "unknown"
                if prompt_class_ids and class_id in prompt_class_ids:
                    idx = prompt_class_ids.index(class_id)
                    class_name = prompt_class_names[idx]

                for instance in instances:
                    if isinstance(instance, (tuple, list)) and len(instance) == 2:
                        mask, score = instance
                    else:
                        mask, score = instance, None

                    mask = mask.astype(bool)
                    if not np.any(mask):
                        continue

                    color = cmap(color_idx % 20)[:3]
                    colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
                    colored_mask[..., :3] = color
                    colored_mask[..., 3] = alpha
                    ax.imshow(colored_mask)

                    ys, xs = np.where(mask)
                    if len(ys) > 0 and len(xs) > 0:
                        x_min, x_max = xs.min(), xs.max()
                        y_min, y_max = ys.min(), ys.max()
                        width, height = x_max - x_min, y_max - y_min

                        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor="none")
                        ax.add_patch(rect)

                        if score is not None:
                            label = f"{class_name} (id={class_id}, conf={score:.2f})"
                        else:
                            label = f"{class_name} (id={class_id})"
                        ax.text(x_min, max(0, y_min - 5), label, color="white", fontsize=10, fontweight="bold",
                                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))

                    color_idx += 1

            ax.set_title(f"Predicted Masks: {os.path.basename(img_path)}")
            ax.axis("off")
            plt.tight_layout()

            output_dir = "."
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"debug_pred_{os.path.basename(img_path)}")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved debug visualization to: {save_path}")
            plt.close(fig)

# ------------------------------------------------------------------------------------------
# --------------------------------------- Experiment ---------------------------------------
# ------------------------------------------------------------------------------------------

def run_experiment(
    image_paths, ann_ids_list, coco, category_names, ignore_ids, gt_class_dict,
    env="gsam", model_name="all", batch_size=8, device="cuda:0",
    prompt_class_ids=None, prompt_class_names=None, void_class_ids=None, confusion_pairs=None,
    metrics_csv_path=None, ap_iou_thresholds=None, instance_class_ids=None, coco_id_remap=None
):
    if env == "gsam":
        import gsam_masks_helper as helper
    elif env == "clipdino":
        import clipdino_masks_helper as helper
    elif env == "radio":
        import radio_masks_helper as helper
    else:
        raise ValueError("Invalid environment specified.")

    print(f"Running {model_name}...")
    model = helper.init_model(model_name, prompt_class_names, device)
    evaluation = None
    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))

        try:
            batch_image_paths = image_paths[start_idx:end_idx]
            batch_ann_ids = ann_ids_list[start_idx:end_idx]

            batch_images, batch_gt_masks, batch_gt_class_ids = [], [], []
            for img_path, ann_ids in zip(batch_image_paths, batch_ann_ids):
                image, masks, class_ids, _ = load_image_and_gt(
                    img_path, ann_ids, coco, category_names, ignore_ids, coco_id_remap
                )
                batch_images.append(image)
                batch_gt_masks.append(masks)
                batch_gt_class_ids.append(class_ids)

            start_time = time.perf_counter()

            batch_masks = helper.predict_batch_masks(
                model=model,
                current_model_name=model_name,
                batch_images=batch_images,
                batch_image_paths=batch_image_paths,
                prompt_class_ids=prompt_class_ids,
                prompt_class_names=prompt_class_names,
                device=device
            )

            runtime = time.perf_counter() - start_time

            TARGET_DEBUG_IMAGE = None
            if TARGET_DEBUG_IMAGE is not None:
                debug_visualize_predictions(
                    batch_images=batch_images,
                    batch_image_paths=batch_image_paths,
                    batch_masks=batch_masks,
                    prompt_class_ids=prompt_class_ids,
                    prompt_class_names=prompt_class_names,
                    target_image_name=TARGET_DEBUG_IMAGE
                )

            evaluation = evaluate_batch(
                batch_masks=batch_masks,
                batch_gt_masks=batch_gt_masks,
                batch_class_ids=batch_gt_class_ids,
                batch_images=batch_images,
                class_ids=prompt_class_ids,
                void_class_ids=void_class_ids,
                accumulator=evaluation,
                runtime=runtime,
                confusion_pairs=confusion_pairs,
                iou_thresholds=ap_iou_thresholds,
                instance_class_ids=instance_class_ids,
                csv_path=metrics_csv_path,
            )

            del batch_images, batch_gt_masks, batch_gt_class_ids, batch_masks

        except Exception as e:
            print(f"WARNING: batch {batch_num + 1}/{num_batches} failed with {type(e).__name__}: {e}")
            continue

    helper.cleanup_model(model)
    return evaluation

def arg_parser():
    parser = ArgumentParser(description="Running Experiment 1: Text Query to Class Mask")
    parser.add_argument('conda_env', type=str, default='gsam', help="Enter conda env framework from [gsam, clipdino, radio]")
    parser.add_argument('--image_dataset', type=str, required=True, help="Folderpath to dataset being evaluated on")
    parser.add_argument('--output', type=str, default='', help="Output folder to save all results to")
    parser.add_argument('--image_subset', type=int, default=-1, help="Only evaluate on <image_subset> number of images")
    parser.add_argument('--model', type=str, default="all", help="Model name to evaluate")
    parser.add_argument('--batch_size', type=int, default=8, help="Image batch size")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run models on")
    parser.add_argument('--ap_iou_thresholds', type=str, default='', help="Comma-separated IoU thresholds for AP")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()

    if args.conda_env not in ["gsam", "clipdino", "radio"]:
        print("Incorrect conda environment name. Should be either [gsam, clipdino, radio]")
        exit()

    output_folder = args.output if len(args.output) > 0 else None
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    if args.ap_iou_thresholds.strip():
        ap_iou_thresholds = tuple(float(t.strip()) for t in args.ap_iou_thresholds.split(","))
    else:
        ap_iou_thresholds = DEFAULT_AP_IOU_THRESHOLDS

    (
        image_paths, ann_ids_list, coco, category_names, ignore_ids,
        gt_class_dict, custom_prompts, void_class_ids, confusion_pairs_yaml,
        instance_class_ids, coco_id_remap, _exp2_top_k, _exp2_min_class_count, _exp2_plot_k
    ) = load_data(
        args.image_dataset,
        image_subset=args.image_subset,
    )

    debug_visualize_random_image(image_paths, ann_ids_list, coco, category_names, ignore_ids, coco_id_remap)

    name_to_id = {v: k for k, v in gt_class_dict.items()}
    prompt_category_dict, prompt_groups = build_prompt_mapping(custom_prompts, name_to_id)
    prompt_class_ids, prompt_class_names = [], []
    for prompt_id, prompts in prompt_groups.items():
        for p in prompts:
            prompt_class_ids.append(prompt_id)
            prompt_class_names.append(p)

    name_to_prompt_id = {v: k for k, v in prompt_category_dict.items()}
    confusion_pairs = []
    for pair in confusion_pairs_yaml:
        if len(pair) == 2:
            id_a = name_to_prompt_id.get(pair[0])
            id_b = name_to_prompt_id.get(pair[1])
            if id_a is not None and id_b is not None:
                confusion_pairs.append((id_a, id_b))
            else:
                print(f"Warning: Confusion pair {pair} not found in custom_prompts targets. Skipping.")

    dataset_name = os.path.basename(args.image_dataset)
    raw_csv_path = (
        os.path.join(output_folder, f"{args.model}_{dataset_name}_raw_progress.csv")
        if output_folder else None
    )

    evaluation_accumulator = None
    try:
        evaluation_accumulator = run_experiment(
            image_paths,
            ann_ids_list,
            coco,
            category_names,
            ignore_ids,
            gt_class_dict,
            env=args.conda_env,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            prompt_class_ids=prompt_class_ids,
            prompt_class_names=prompt_class_names,
            void_class_ids=void_class_ids,
            confusion_pairs=confusion_pairs,
            metrics_csv_path=raw_csv_path,
            ap_iou_thresholds=ap_iou_thresholds,
            instance_class_ids=instance_class_ids,
            coco_id_remap=coco_id_remap
        )
    finally:
        if evaluation_accumulator is not None:
            if output_folder:
                final_results = finalize_evaluation(
                    evaluation_accumulator,
                    prompt_category_dict,
                    csv_path=os.path.join(output_folder, f"{args.model}_{dataset_name}_results.csv"),
                )
            else:
                final_results = finalize_evaluation(
                    evaluation_accumulator,
                    prompt_category_dict,
                )
            print_evaluation_results(final_results, args.model, dataset_name)