from argparse import ArgumentParser
import os
import glob
import random
import time
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pycocotools.coco import COCO

# Import metrics logic originally held in the helper
from metrics import (
    evaluate_batch,
    finalize_evaluation,
    print_evaluation_results,
)

# ------------------------------------------------------------------------------------------
# --------------------------------------- Load Data ----------------------------------------
# ------------------------------------------------------------------------------------------

def load_data(dataset_dir, annotation_file=None, image_subset=-1):
    """Load COCO annotations, YAML config, and return dataset and parameters."""

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
    custom_prompts = config.get("custom_prompts", None)
    confusion_pairs_yaml = config.get("confusion_pairs", [])

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
    
    # 3. Process Ignore and Void Classes (Strings to IDs)
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

    gt_class_dict = {k: v for k, v in category_names.items() if k not in ignore_ids}

    # 4. Standardize Custom Prompts Mapping
    if custom_prompts is None:
        # Default to ground truth classes
        custom_prompts = {
            name: name for class_id, name in gt_class_dict.items() 
            if class_id not in void_class_ids
        }
    elif isinstance(custom_prompts, dict):
        # Handle dict format where some values might be None (blank in YAML)
        for k, v in custom_prompts.items():
            if v is None:
                custom_prompts[k] = k
    elif isinstance(custom_prompts, list):
        # Handle list format with a mix of dicts and strings
        merged_prompts = {}
        for item in custom_prompts:
            if isinstance(item, dict):
                for k, v in item.items():
                    merged_prompts[k] = v if v is not None else k
            elif isinstance(item, str):
                merged_prompts[item] = item
        custom_prompts = merged_prompts

    # 5. Collect image metadata only (NOT decoded pixels/masks).
    # Decoding every image + every instance mask for the whole dataset upfront
    # is what was blowing up memory on larger datasets (e.g. LaRS) -- it defeats
    # the point of the batch_size used later. We now just record enough to
    # decode each image lazily, per-batch, in run_experiment().
    image_ids = coco.getImgIds()
    if image_subset > 0:
        image_ids = image_ids[:image_subset]

    image_paths = []
    ann_ids_list = []  # ann ids per image, decoded on demand later

    for image_id in image_ids:
        img_meta = coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset_dir, img_meta["file_name"])
        image_paths.append(image_path)
        ann_ids_list.append(coco.getAnnIds(imgIds=[image_id]))

    return (
        image_paths, ann_ids_list, coco, category_names, ignore_ids,
        gt_class_dict, custom_prompts, void_class_ids, confusion_pairs_yaml,
    )

# ------------------------------------------------------------------------------------------
# --------------------------------------- Lazy Decode --------------------------------------
# ------------------------------------------------------------------------------------------

def load_image_and_gt(image_path, ann_ids, coco, category_names, ignore_ids):
    """
    Decode a single image + its ground-truth instance masks on demand.
    This is the piece that used to run eagerly, for every image, inside load_data().
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    annotations = coco.loadAnns(ann_ids)

    image_masks, image_class_ids, image_class_names = [], [], []

    for ann in annotations:
        category_id = int(ann["category_id"])
        if category_id in ignore_ids:
            continue

        mask = coco.annToMask(ann)
        mask = (mask > 0).astype(np.uint8)

        image_masks.append(mask)
        image_class_ids.append(category_id)
        image_class_names.append(category_names.get(category_id, "unknown"))

    return image, image_masks, image_class_ids, image_class_names

# ------------------------------------------------------------------------------------------
# ----------------------------------------- Debug ------------------------------------------
# ------------------------------------------------------------------------------------------

def debug_visualize_random_image(image_paths, ann_ids_list, coco, category_names, ignore_ids, alpha=0.45):
    """Randomly visualize one image with instance masks and bounding boxes.

    Decodes only the single chosen image on demand instead of requiring the
    whole dataset to already be loaded into memory.
    """

    if len(image_paths) == 0:
        print("No images available.")
        return

    image_idx = random.randrange(len(image_paths))
    image, image_masks, image_class_ids, image_class_names = load_image_and_gt(
        image_paths[image_idx], ann_ids_list[image_idx], coco, category_names, ignore_ids
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
            print(f"WARNING: Mask {mask_idx} is empty.")
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
    """Visualize predicted masks for a specific target image from a batch."""
    
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
            for class_id, masks in pred_masks_dict.items():
                # Attempt to resolve the prompt name for the label
                class_name = "unknown"
                if prompt_class_ids and class_id in prompt_class_ids:
                    idx = prompt_class_ids.index(class_id)
                    class_name = prompt_class_names[idx]

                for mask in masks:
                    mask = mask.astype(bool)
                    if not np.any(mask):
                        continue
                    
                    color = cmap(color_idx % 20)[:3]
                    
                    # Apply colored mask
                    colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
                    colored_mask[..., :3] = color
                    colored_mask[..., 3] = alpha
                    ax.imshow(colored_mask)
                    
                    # Generate Bounding Box
                    ys, xs = np.where(mask)
                    if len(ys) > 0 and len(xs) > 0:
                        x_min, x_max = xs.min(), xs.max()
                        y_min, y_max = ys.min(), ys.max()
                        width, height = x_max - x_min, y_max - y_min
                        
                        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor="none")
                        ax.add_patch(rect)
                        
                        # Add text label
                        label = f"{class_name} (id={class_id})"
                        ax.text(x_min, max(0, y_min - 5), label, color="white", fontsize=10, fontweight="bold", 
                                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))
                    
                    color_idx += 1
                    
            ax.set_title(f"Predicted Masks: {os.path.basename(img_path)}")
            ax.axis("off")
            plt.tight_layout()

            # Save the figure instead of trying to show it
            output_dir = "."
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"debug_pred_{os.path.basename(img_path)}")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved debug visualization to: {save_path}")
            
            # Close the figure to free up memory
            plt.close(fig)

# ------------------------------------------------------------------------------------------
# --------------------------------------- Experiment ---------------------------------------
# ------------------------------------------------------------------------------------------

def run_experiment(
    image_paths, ann_ids_list, coco, category_names, ignore_ids, gt_class_dict,
    env="gsam", model_name="all", batch_size=8, device="cuda:0",
    prompt_class_ids=None, prompt_class_names=None, void_class_ids=None, confusion_pairs=None,
    metrics_csv_path=None,
):
    
    # --------------------------------------------------
    # Environment Setup
    # --------------------------------------------------
    if env == "gsam":
        import gsam_masks_helper as helper
    elif env == "clipdino":
        import clipdino_masks_helper as helper
    elif env == "radio":
        import radio_masks_helper as helper
    else:
        raise ValueError("Invalid environment specified.")

    print(f"Running {model_name}...")

    # --------------------------------------------------
    # Model Setup
    # --------------------------------------------------

    # 1. Initialize the model once
    model = helper.init_model(model_name, prompt_class_names, device)
    
    evaluation = None

    # --------------------------------------------------
    # Experiment Loop
    # --------------------------------------------------

    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    # 2. Main script controls the batch loop
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))

        try:
            batch_image_paths = image_paths[start_idx:end_idx]
            batch_ann_ids = ann_ids_list[start_idx:end_idx]

            # Decode this batch's images + GT masks now, not upfront for the
            # whole dataset. Keeps peak memory to ~batch_size images at a
            # time regardless of how large the dataset is.
            batch_images, batch_gt_masks, batch_gt_class_ids = [], [], []
            for img_path, ann_ids in zip(batch_image_paths, batch_ann_ids):
                image, masks, class_ids, _ = load_image_and_gt(
                    img_path, ann_ids, coco, category_names, ignore_ids
                )
                batch_images.append(image)
                batch_gt_masks.append(masks)
                batch_gt_class_ids.append(class_ids)

            start_time = time.perf_counter()

            # 3. Ask helper for predictions
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

            # --------------------------------------------------
            # Debugger
            # --------------------------------------------------

            # change variable to name of image
            TARGET_DEBUG_IMAGE = None

            if TARGET_DEBUG_IMAGE is not None:
                # dispaly predicted masks
                debug_visualize_predictions(
                    batch_images=batch_images,
                    batch_image_paths=batch_image_paths,
                    batch_masks=batch_masks,
                    prompt_class_ids=prompt_class_ids,
                    prompt_class_names=prompt_class_names,
                    target_image_name=TARGET_DEBUG_IMAGE
                )

            # --------------------------------------------------
            # Evaluation
            # --------------------------------------------------

            # 4. Evaluate immediately
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
                # Write raw TP/FP/FN/TN after every batch. If the process gets
                # killed partway through a large dataset, this file still has
                # everything evaluated up to that point instead of nothing.
                csv_path=metrics_csv_path,
            )

            # Explicitly drop this batch's decoded arrays before moving on.
            del batch_images, batch_gt_masks, batch_gt_class_ids, batch_masks

        except Exception as e:
            # Don't let one bad batch (a corrupt image, a transient CUDA OOM,
            # etc.) throw away every batch that was already evaluated.
            # `evaluation` still holds everything accumulated so far, and
            # metrics_csv_path (if set) already reflects it on disk.
            print(
                f"WARNING: batch {batch_num + 1}/{num_batches} "
                f"(images {start_idx}:{end_idx}) failed with {type(e).__name__}: {e}"
            )
            print("Skipping this batch and continuing with the next one.")
            continue

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    # 5. Clean up the model to free GPU memory
    helper.cleanup_model(model)

    return evaluation

def arg_parser():
    parser = ArgumentParser(description="Running Experiment 1: Text Query to Class Mask")
    parser.add_argument('conda_env', type=str, default='gsam', help="Enter corresponding conda env for dependency purposes from [gsam, clipdino, radio]")
    parser.add_argument('--image_dataset', type=str, required=True, help="Folderpath to dataset being evaluated on")
    parser.add_argument('--output', type=str, default='', help="Output folder to save all results to")
    parser.add_argument('--image_subset', type=int, default=-1, help="(Optional) Only evaluate on <image_subset> number of images")
    parser.add_argument('--model', type=str, default="all", help="(Optional) Name of VLM model to evaluate if not all [yoloe, clip, siglip, gsam, sam3, radio, clipdino]")
    parser.add_argument('--batch_size', type=int, default=8, help="(Optional) Image batch size")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run models on [cuda:0, cpu]")
    return parser.parse_args()

if __name__ == "__main__":

    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------
    args = arg_parser()
    
    if args.conda_env not in ["gsam", "clipdino", "radio"]:
        print("Incorrect conda environment name. Should be either [gsam, clipdino, radio]") 
        exit()
    
    output_folder = args.output if len(args.output) > 0 else None
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------
    
    (
        image_paths, ann_ids_list, coco, category_names, ignore_ids,
        gt_class_dict, custom_prompts, void_class_ids, confusion_pairs_yaml,
    ) = load_data(
        args.image_dataset,
        image_subset=args.image_subset,
    )

    debug_visualize_random_image(image_paths, ann_ids_list, coco, category_names, ignore_ids)

    # --------------------------------------------------
    # Input
    # --------------------------------------------------
    
    # Process custom prompts mapping directly here
    ### prompts with GT associations receive positive ids
    ### prompts without GT associations receive negative ids (skipped in metrics)
    name_to_id = {v: k for k, v in gt_class_dict.items()}
    prompt_category_dict = {}
    prompt_class_ids = []
    prompt_class_names = []

    neg_id_counter = -1
    
    # Iterate through the parsed dictionary keys and values
    for target_class, prompt_string in custom_prompts.items():
        if target_class in name_to_id:
            prompt_id = name_to_id[target_class]
        else:
            prompt_id = neg_id_counter
            neg_id_counter -= 1
        
        prompt_category_dict[prompt_id] = target_class
        
        # Split the string by commas and register each specific prompt
        prompts = [p.strip() for p in str(prompt_string).split(",")]
        for p in prompts:
            if not p:
                continue
            prompt_class_ids.append(prompt_id)
            prompt_class_names.append(p)

    # --------------------------------------------------
    # Process Confusion Pairs
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Process
    # --------------------------------------------------

    dataset_name = os.path.basename(args.image_dataset)

    # Raw per-batch accumulator CSV (TP/FP/FN/TN only, no derived metrics yet).
    # Gets overwritten after every batch, so if the run dies partway through
    # a large dataset, this file still reflects everything evaluated so far.
    raw_csv_path = (
        os.path.join(output_folder, f"{args.model}_{dataset_name}_raw_progress.csv")
        if output_folder else None
    )

    evaluation_accumulator = None
    try:
        # Execute experiment batch loop and fetch evaluation accumulator
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
        )
    finally:
        # Even on an exception (NOT a hard OOM kill -- nothing can catch
        # SIGKILL -- but any normal Python-level error), still write out
        # whatever was accumulated so far rather than losing it silently.
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