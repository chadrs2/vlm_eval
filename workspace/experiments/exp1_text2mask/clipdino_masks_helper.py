import os, sys
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn.functional as F
import numpy as np
from operator import itemgetter 
import torch
import gc
import cv2
import warnings
warnings.filterwarnings('ignore')

CLIPDINO_PROJECT_ROOT = "/workspace/projects/clip_dinoiser"
CLIPDINO_CONFIG_DIR = os.path.join(CLIPDINO_PROJECT_ROOT, "configs")
CLIPDINO_CHECKPOINT = os.path.join(CLIPDINO_PROJECT_ROOT, "checkpoints", "last.pt")
if CLIPDINO_PROJECT_ROOT not in sys.path:
    sys.path.append(CLIPDINO_PROJECT_ROOT)
from models.builder import build_model
from segmentation.datasets import PascalVOCDataset
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import time
from metrics import (
    evaluate_batch,
    finalize_evaluation,
    print_evaluation_results,
)


def load_clipdino(
    all_class_names,
    device,
    config_dir=CLIPDINO_CONFIG_DIR,
    config_name="clip_dinoiser.yaml",
    checkpoint_path=CLIPDINO_CHECKPOINT,
):
    # (Re-)initialize Hydra pointed at the project's config dir. Guard
    # against "already initialized" errors when this is called more than
    # once in the same process (e.g. across multiple experiment runs).
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name=config_name)
 
    # Build model + load checkpoint while cwd is the project root so the
    # config's relative paths resolve correctly, then restore cwd.
    orig_cwd = os.getcwd()
    try:
        os.chdir(CLIPDINO_PROJECT_ROOT)
        check = torch.load(checkpoint_path, map_location="cpu")
        model = build_model(cfg.model, class_names=PascalVOCDataset.CLASSES).to(device)
    finally:
        os.chdir(orig_cwd)
 
    model.clip_backbone.decode_head.use_templates = False  # faster inference
    model.load_state_dict(check["model_state_dict"], strict=False)
    model = model.eval()
    return model
 
 
# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_clipdino(
    model,
    images,
    gt_class_ids,
    gt_class_names,
    device,
    apply_found=True,
):
    """
    Runs CLIP-DINOiser dense per-pixel classification over a batch of
    images and converts the argmax class map into per-class binary masks,
    following the same batch_masks dict-of-lists structure used by the
    other predict_* functions (gt_class_id -> [mask_np, ...]).
 
    When apply_found=True, CLIP-DINOiser's FOUND background detector needs
    an explicit "background" class at vocab index 0 (mirroring the
    notebook usage), so it is injected automatically and is not itself
    returned as one of the gt classes. When apply_found=False, the raw
    class list is used as the vocab and every pixel is forced into one of
    the given prompts (no separate background/void class).
    """
    vocab = (["background"] + list(gt_class_names)) if apply_found else list(gt_class_names)
    class_offset = 1 if apply_found else 0
 
    model.clip_backbone.decode_head.update_vocab(vocab)
    model.apply_found = apply_found
    model.to(device)
 
    batch_masks = []
    for image in images:
        IMG_H, IMG_W = image.shape[0], image.shape[1]
 
        # numpy HWC uint8 image -> normalized CHW tensor on device
        img_tens = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        img_tens = img_tens.permute(2, 0, 1).unsqueeze(0) / 255.0
 
        h, w = img_tens.shape[-2:]
        with torch.no_grad():
            output = model(img_tens).cpu()
        output = F.interpolate(
            output,
            scale_factor=model.vit_patch_size,
            mode="bilinear",
            align_corners=False,
        )[..., :h, :w]
        class_map = output[0].argmax(dim=0).numpy()  # H x W vocab-index map
 
        masks = {}
        for c_idx, class_name in enumerate(gt_class_names):
            gt_class_id = gt_class_ids[c_idx]
            vocab_idx = c_idx + class_offset
 
            binary_mask = (class_map == vocab_idx)
            mask_np = (binary_mask.astype(np.uint8) * 255)  # Convert mask to 0-255
            mask_resized = cv2.resize(
                mask_np,
                (IMG_W, IMG_H),
                interpolation=cv2.INTER_NEAREST,
            )
 
            if gt_class_id in masks:
                masks[gt_class_id].append(mask_resized)
            else:
                masks[gt_class_id] = [mask_resized]
 
        batch_masks.append(masks)
 
    return batch_masks
 
 
# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment1(
    images,
    image_paths,
    gt_masks,
    class_ids,
    class_names,
    category_name_dict,
    model_name,
    output_folder,
    batch_size=8,
    device="cuda:0",
    custom_prompts=None,
    void_class_ids=None,
    apply_found=True,
):
 
    if custom_prompts is None:
        custom_prompts = ["boat", "person", "shore"]
 
    name_to_id = {v: k for k, v in category_name_dict.items()}
    eval_category_dict = {}
    eval_class_ids = []
    eval_class_names = []
 
    # Map prompts to GT IDs, generating negative IDs for prompts without GT
    neg_id_counter = -1
    for prompt in custom_prompts:
        if prompt in name_to_id:
            prompt_id = name_to_id[prompt]
        else:
            prompt_id = neg_id_counter
            neg_id_counter -= 1
 
        eval_category_dict[prompt_id] = prompt
        eval_class_ids.append(prompt_id)
        eval_class_names.append(prompt)
 
 
    model = load_clipdino(eval_class_names, device)

    print(f"Running clip-dinoiser...")

    evaluation = None

    for start_idx in range(0, len(images), batch_size):

        end_idx = min(
            start_idx + batch_size,
            len(images),
        )

        batch_images = images[start_idx:end_idx]
        batch_image_paths = image_paths[start_idx:end_idx]
        batch_gt_masks = gt_masks[start_idx:end_idx]
        batch_class_ids = class_ids[start_idx:end_idx]

        start_time = time.perf_counter()
        batch_masks = predict_clipdino(
            model,
            batch_images,
            eval_class_ids,
            eval_class_names,
            device,
            apply_found=apply_found,
        )
        runtime = time.perf_counter() - start_time

        # Evaluate batch masks
        evaluation = evaluate_batch(
            batch_masks=batch_masks,
            batch_gt_masks=batch_gt_masks,
            batch_class_ids=batch_class_ids,
            batch_images=batch_images,
            class_ids=eval_class_ids,
            void_class_ids=void_class_ids,
            accumulator=evaluation,
            runtime=runtime,
        )

    # ---------------------------------------------------------
    # Finalize metrics over entire dataset
    # ---------------------------------------------------------
    if output_folder:
        results = finalize_evaluation(
            evaluation,
            eval_category_dict,
            csv_path=os.path.join(output_folder, f"{model_name}_final_results.csv"),
        )
    else:
        results = finalize_evaluation(
            evaluation,
            eval_category_dict,
        )

    print_evaluation_results(results)

    # --- cleanup before loading the next model ---
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
 