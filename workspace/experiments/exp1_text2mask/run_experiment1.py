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
        
    ignore_classes = config.get("ignore_classes", [])
    void_classes = config.get("void_classes", [])
    custom_prompts = config.get("custom_prompts", None)

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

    # Default to ground truth classes (as a dict) if custom_prompts is None
    if custom_prompts is None:
        custom_prompts = {
            name: name for class_id, name in gt_class_dict.items() 
            if class_id not in void_class_ids
        }
    elif isinstance(custom_prompts, list):
        # Handle YAML list of dictionaries (e.g., "- key: value")
        if all(isinstance(item, dict) for item in custom_prompts):
            merged_prompts = {}
            for d in custom_prompts:
                merged_prompts.update(d)
            custom_prompts = merged_prompts
        else:
            # Fallback for a flat list of strings (e.g., "- shore-artificial")
            custom_prompts = {p: p for p in custom_prompts}

    # 4. Load Images and Masks
    image_ids = coco.getImgIds()
    if image_subset > 0:
        image_ids = image_ids[:image_subset]

    images, image_paths, gt_masks, gt_class_ids, gt_class_names = [], [], [], [], []

    for image_id in image_ids:
        img_meta = coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset_dir, img_meta["file_name"])
        image = np.array(Image.open(image_path).convert("RGB"))
        
        image_paths.append(image_path)
        image_masks, image_class_ids, image_class_names = [], [], []

        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            category_id = int(ann["category_id"])
            if category_id in ignore_ids:
                continue

            mask = coco.annToMask(ann)
            mask = (mask > 0).astype(np.uint8)

            image_masks.append(mask)
            image_class_ids.append(category_id)
            image_class_names.append(category_names.get(category_id, "unknown"))

        images.append(image)
        gt_masks.append(image_masks)
        gt_class_ids.append(image_class_ids)
        gt_class_names.append(image_class_names)

    return images, image_paths, gt_masks, gt_class_ids, gt_class_names, gt_class_dict, custom_prompts, void_class_ids

# ------------------------------------------------------------------------------------------
# ----------------------------------------- Debug ------------------------------------------
# ------------------------------------------------------------------------------------------

def debug_visualize_random_image(images, masks, class_ids, class_names, alpha=0.45):
    """Randomly visualize one image with instance masks and bounding boxes."""

    if len(images) == 0:
        print("No images available.")
        return

    image_idx = random.randrange(len(images))
    image = images[image_idx]
    image_masks = masks[image_idx]
    image_class_ids = class_ids[image_idx]
    image_class_names = class_names[image_idx]

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

# ------------------------------------------------------------------------------------------
# --------------------------------------- Experiment ---------------------------------------
# ------------------------------------------------------------------------------------------

def run_experiment(
    images, image_paths, gt_masks, gt_class_ids, gt_class_names, gt_class_dict, 
    env="gsam", model_name="all", batch_size=8, device="cuda:0",
    prompt_class_ids=None, prompt_class_names=None, void_class_ids=None
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

    # 2. Main script controls the batch loop
    for start_idx in range(0, len(images), batch_size):
        end_idx = min(start_idx + batch_size, len(images))
        
        batch_images = images[start_idx:end_idx]
        batch_image_paths = image_paths[start_idx:end_idx]
        batch_gt_masks = gt_masks[start_idx:end_idx]
        batch_gt_class_ids = gt_class_ids[start_idx:end_idx]
        
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
        )

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
    
    images, image_paths, gt_masks, gt_class_ids, gt_class_names, gt_class_dict, custom_prompts, void_class_ids = load_data(
        args.image_dataset,
        image_subset=args.image_subset,
    )

    debug_visualize_random_image(images, gt_masks, gt_class_ids, gt_class_names)

    # --------------------------------------------------
    # Input
    # --------------------------------------------------
    
    # Process custom prompts mapping directly here
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
    # Process
    # --------------------------------------------------

    # Execute experiment batch loop and fetch evaluation accumulator
    evaluation_accumulator = run_experiment(
        images, 
        image_paths,
        gt_masks, 
        gt_class_ids, 
        gt_class_names,
        gt_class_dict,
        env=args.conda_env, 
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        prompt_class_ids=prompt_class_ids,
        prompt_class_names=prompt_class_names,
        void_class_ids=void_class_ids
    )
    
    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    if output_folder:
        final_results = finalize_evaluation(
            evaluation_accumulator,
            prompt_category_dict,
            csv_path=os.path.join(output_folder, f"{args.model}_final_results.csv"),
        )
    else:
        final_results = finalize_evaluation(
            evaluation_accumulator,
            prompt_category_dict,
        )

    print_evaluation_results(final_results)