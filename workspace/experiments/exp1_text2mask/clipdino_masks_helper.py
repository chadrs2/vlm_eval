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

# ---------------------------------------------------------------------------
# Internal Functions
# ---------------------------------------------------------------------------

def _load_clipdino(
    all_class_names,
    device,
    config_dir=CLIPDINO_CONFIG_DIR,
    config_name="clip_dinoiser.yaml",
    checkpoint_path=CLIPDINO_CHECKPOINT,
):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name=config_name)
 
    orig_cwd = os.getcwd()
    try:
        os.chdir(CLIPDINO_PROJECT_ROOT)
        check = torch.load(checkpoint_path, map_location="cpu")
        model = build_model(cfg.model, class_names=PascalVOCDataset.CLASSES).to(device)
    finally:
        os.chdir(orig_cwd)
 
    model.clip_backbone.decode_head.use_templates = False  
    model.load_state_dict(check["model_state_dict"], strict=False)
    model = model.eval()
    return model
 
def _predict_clipdino(model, images, prompt_class_ids, prompt_class_names, device, apply_found=True):

    vocab = (["background"] + list(prompt_class_names)) if apply_found else list(prompt_class_names)
    class_offset = 1 if apply_found else 0
 
    model.clip_backbone.decode_head.update_vocab(vocab)
    model.apply_found = apply_found
    model.to(device)
 
    batch_masks = []
    for image in images:
        IMG_H, IMG_W = image.shape[0], image.shape[1]
 
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
        class_map = output[0].argmax(dim=0).numpy()  
 
        masks = {}
        for c_idx, class_name in enumerate(prompt_class_names):
            gt_class_id = prompt_class_ids[c_idx]
            vocab_idx = c_idx + class_offset
 
            binary_mask = (class_map == vocab_idx)
            mask_np = (binary_mask.astype(np.uint8) * 255)  
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

# ------------------------------------------------------------------------------------------
# ----------------------------- Helper API (Exposed Functions) -----------------------------
# ------------------------------------------------------------------------------------------

def init_model(model_name, prompt_class_names, device):
    """Loads and returns the CLIP-DINOiser model."""
    return _load_clipdino(prompt_class_names, device)

def predict_batch_masks(model, current_model_name, batch_images, batch_image_paths, prompt_class_ids, prompt_class_names, device):
    """Generates masks for a batch of images using CLIP-DINOiser."""
    return _predict_clipdino(model, batch_images, prompt_class_ids, prompt_class_names, device, apply_found=True)

def cleanup_model(model):
    """Frees up GPU memory after all batches are done."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()