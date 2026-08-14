from argparse import ArgumentParser
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pycocotools.coco import COCO


def load_data(dataset_dir, class_ids_ignore=None, annotation_file=None, image_subset=-1):
    """Load COCO annotations and return images, masks, and file paths.

    Returns: images, masks, class_ids, class_names, category_name_dict, image_paths
    """

    dataset_dir = os.path.abspath(dataset_dir)

    if annotation_file is None:
        annotation_file = os.path.join(
            dataset_dir,
            "_annotations.coco.json"
        )

    ignore_ids = set(class_ids_ignore or [])

    # Load COCO annotations
    coco = COCO(annotation_file)

    # Category ID -> category name
    categories = coco.loadCats(coco.getCatIds())
    category_names = {
        cat["id"]: cat["name"]
        for cat in categories
    }
    category_name_dict = {k: v for k, v in category_names.items() if k not in ignore_ids}

    # Get image IDs
    image_ids = coco.getImgIds()

    if image_subset > 0:
        image_ids = image_ids[:image_subset]

    images = []
    masks = []
    class_ids = []
    class_names = []
    image_paths = []

    for image_id in image_ids:

        # Image metadata
        img_meta = coco.loadImgs(image_id)[0]

        image_path = os.path.join(
            dataset_dir,
            img_meta["file_name"]
        )

        image = np.array(
            Image.open(image_path).convert("RGB")
        )
        # Track the original image filepath
        image_paths.append(image_path)

        image_masks = []
        image_class_ids = []
        image_class_names = []

        # Get all annotations for this image
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:

            category_id = int(ann["category_id"])

            # Ignore unwanted classes
            if category_id in ignore_ids:
                continue

            # Convert COCO polygon/RLE -> binary mask
            mask = coco.annToMask(ann)

            # Ensure uint8 binary mask
            mask = (mask > 0).astype(np.uint8)

            image_masks.append(mask)
            image_class_ids.append(category_id)
            image_class_names.append(
                category_names.get(category_id, "unknown")
            )

        images.append(image)
        masks.append(image_masks)
        class_ids.append(image_class_ids)
        class_names.append(image_class_names)

    return images, masks, class_ids, class_names, category_name_dict, image_paths


def debug_visualize_random_image(
    images,
    masks,
    class_ids,
    class_names,
    alpha=0.45,
):
    """Randomly visualize one image with instance masks and bounding boxes."""

    if len(images) == 0:
        print("No images available.")
        return

    # Randomly select an image
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

    # Generate distinct colors for each mask
    cmap = plt.get_cmap("tab20")

    for mask_idx, mask in enumerate(image_masks):

        # Make sure mask is binary
        mask = mask.astype(bool)

        if not np.any(mask):
            print(f"WARNING: Mask {mask_idx} is empty.")
            continue

        # ---------------------------------------------------------
        # Mask overlay
        # ---------------------------------------------------------
        color = cmap(mask_idx % 20)[:3]

        colored_mask = np.zeros(
            (*mask.shape, 4),
            dtype=np.float32
        )

        colored_mask[..., :3] = color
        colored_mask[..., 3] = mask * alpha

        ax.imshow(colored_mask)

        # ---------------------------------------------------------
        # Bounding box
        # ---------------------------------------------------------
        ys, xs = np.where(mask)

        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )

        ax.add_patch(rect)

        # ---------------------------------------------------------
        # Label
        # ---------------------------------------------------------
        class_name = image_class_names[mask_idx]
        class_id = image_class_ids[mask_idx]

        label = f"{mask_idx}: {class_name} (id={class_id})"

        ax.text(
            x_min,
            max(0, y_min - 5),
            label,
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                facecolor=color,
                alpha=0.8,
                pad=2,
                edgecolor="none",
            ),
        )

    ax.set_title(
        f"Image {image_idx} — {len(image_masks)} masks"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.show()

def run_experiment(
    images, 
    image_paths,
    gt_masks, 
    class_ids, 
    class_names, 
    category_name_dict, 
    env="gsam", 
    model="all", 
    output_folder=None,
    batch_size=8,
    device="cuda:0",
    custom_prompts=None,
    void_class_ids=None
):
    # Import conda env variables
    if env == "gsam":
        import gsam_masks_helper
        from gsam_masks_helper import run_experiment1
        run_experiment1(
            images,
            image_paths,
            gt_masks,
            class_ids,
            class_names,
            category_name_dict,
            model_name=model,
            output_folder=output_folder,
            batch_size=batch_size,
            device=device,
            custom_prompts=custom_prompts,
            void_class_ids=void_class_ids
        ) 
    elif env == "clipdino":
        import clipdino_masks_helper
        from clipdino_masks_helper import run_experiment1
        run_experiment1(
            images,
            image_paths,
            gt_masks,
            class_ids,
            class_names,
            category_name_dict,
            model_name=model,
            output_folder=output_folder,
            batch_size=batch_size,
            device=device,
            custom_prompts=custom_prompts,
            void_class_ids=void_class_ids
        )    
    else: # "radio"
        import radio_masks_helper
        from radio_masks_helper import run_experiment1
        run_experiment1(
            images,
            image_paths,
            gt_masks,
            class_ids,
            class_names,
            category_name_dict,
            model_name=model,
            output_folder=output_folder,
            batch_size=batch_size,
            device=device,
            custom_prompts=custom_prompts,
            void_class_ids=void_class_ids
        )    
    pass


def arg_parser():
    parser = ArgumentParser(description="Running Experiment 1: Text Query to Class Mask")
    parser.add_argument('conda_env', 
                        type=str, 
                        default='gsam',
                        help="Enter corresponding conda env for dependency purposes from [gsam, clipdino, radio]")
    parser.add_argument('--image_dataset',
                        type=str,
                        required=True,
                        help="Folderpath to dataset being evaluated on")
    parser.add_argument('--output',
                        type=str,
                        default='',
                        help="Output folder to save all results to")
    parser.add_argument('--image_subset',
                        type=int,
                        default=-1,
                        help="(Optional) Only evaluate on <image_subset> number of images")
    parser.add_argument('--model',
                        type=str,
                        default="all",
                        help="(Optional) Name of VLM model to evaluate if not all [yoloe, clip, siglip, gsam, sam3, radio, clipdino]")
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help="(Optional) Image batch size")
    parser.add_argument('--device',
                        type=str,
                        default="cuda:0",
                        help="Device to run models on [cuda:0, cpu]")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    
    # Import conda env variables
    if args.conda_env != "gsam" and args.conda_env != "clipdino" and args.conda_env != "radio":
        print("Incorrect conda environment name. Should be either [gsam, clipdino, radio]") 
        print("Exiting...")
        exit()
    
    output_folder = args.output
    if len(output_folder) == 0:
        output_folder = None
    else:
        os.makedirs(output_folder,exist_ok=True)
      
    # ---------------------------------------------------------
    # Custom Evaluation Settings
    # ---------------------------------------------------------
    
    void_class_ids = []
    # void_class_ids = [1, 5] # test IDs
    # void_class_ids = [1, 6] # full Hawaii dataset IDS
    
    # Load data
    # (Do NOT ignore 1 and 6 during load_data, so their ground truth is available for void subtraction!)
    # class_ids_ignore = [0] # Hawaii-Database-official 
    class_ids_ignore = [0,1,5] # Hawaii-Database-official 
    
    custom_prompts = [
        "bridge",
        "building",
        "car",
        "mountain",
        "person",
        "pier",
        "shore-artificial",
        "sky",
        "vegetation",
        "water",  
    ]
    # custom_prompts = [
    #     "vegetation",
    #     "boat", 
    #     "car",
    #     "person",
    #     "buoy", "building",
    #     "piling",
    #     "water", "mountain", "sky",
    #     "gangway", "bridge",
    #     "float", "pier", "wharf",
    #     "shore-natural", "shore-artificial"
    # ]
    
    images, masks, class_ids, class_names, cat_name_dict, image_paths = load_data(
        args.image_dataset,
        class_ids_ignore=class_ids_ignore,
        image_subset=args.image_subset,
    )
    # Debug visualization to make sure data is loaded properly
    debug_visualize_random_image(
        images,
        masks,
        class_ids,
        class_names,
    )
    
    # Execute experiment & save results
    results = run_experiment(
        images, 
        image_paths,
        masks, 
        class_ids, 
        class_names,
        cat_name_dict,
        env=args.conda_env, 
        model=args.model,
        output_folder=output_folder,
        batch_size=args.batch_size,
        device=args.device,
        custom_prompts=custom_prompts,
        void_class_ids=void_class_ids
    )
    
    # TODO: Display results