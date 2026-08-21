import os
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoTokenizer, Siglip2TextModel, AutoConfig
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
PROJECTS_ROOT = Path("/workspace/projects")
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

try:
    from nvidia_radio.hubconf import radio_model
    from nvidia_radio.radio.pamr import PAMR
except ImportError:
    from projects.nvidia_radio.hubconf import radio_model
    from projects.nvidia_radio.radio.pamr import PAMR
   
import time
import gc

# ---------------------------------------------------------------------------
# Internal Functions
# ---------------------------------------------------------------------------

def _load_radio(
    all_class_names,
    device,
    model_version="c-radio_v3-b",
    adaptor_version="siglip2",
):
    print(f"Loading {model_version} with {adaptor_version} adapter...")
    model, chk = radio_model(
        version=model_version,
        progress=True,
        skip_validation=True,
        adaptor_names=adaptor_version,
        return_checkpoint=True,
        use_naclip=True,
        naclip_strategy="kkonly",
        naclip_gaussian_std=5.0,
        fixed_patch_dim=(40, 40),
        gaussian_device=device,
        use_summary_for_spatial=True,
    )
    model = model.to(device=device).eval()
    return model

def _predict_radio(model, images, prompt_class_ids, prompt_class_names, device,adaptor_name="siglip2", threshold=0.6):

    adaptor = model.adaptors[adaptor_name]
    text_input = adaptor.tokenizer(prompt_class_names).to(device)
    with torch.no_grad():
        text_tokens = adaptor.encode_text(text_input, normalize=True) 
 
    batch_masks = []
    for image in images:
        IMG_H, IMG_W = image.shape[0], image.shape[1]
 
        image_tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor.div_(255.0)  
 
        nearest_res = model.get_nearest_supported_resolution(*image_tensor.shape[-2:])
        image_tensor = F.interpolate(
            image_tensor, nearest_res, mode="bilinear", align_corners=False
        )
        feat_h = nearest_res[0] // model.patch_size
        feat_w = nearest_res[1] // model.patch_size
 
        with torch.no_grad():
            vis_output = model(image_tensor)
        s2_sum, spatial_vision_features = vis_output[adaptor_name]
 
        spatial_feats = spatial_vision_features.squeeze(0)  
        spatial_feats = spatial_feats / spatial_feats.norm(dim=0, keepdim=True)
 
        similarity = torch.matmul(text_tokens, spatial_feats.T)  
 
        masks = {}
        for c_idx, class_name in enumerate(prompt_class_names):
            gt_class_id = prompt_class_ids[c_idx]
 
            heatmap = similarity[c_idx].view(feat_h, feat_w).detach().cpu().numpy()
            heatmap_resized = cv2.resize(
                heatmap,
                (IMG_W, IMG_H),
                interpolation=cv2.INTER_LINEAR,
            )
 
            h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
            heatmap_resized = (heatmap_resized - h_min) / (h_max - h_min + 1e-8)
 
            binary_mask = heatmap_resized > threshold
            mask_np = (binary_mask.astype(np.uint8) * 255)  
            
            # Calculate instance score as mean confidence within the generated heatmap mask
            if binary_mask.any():
                score = float(heatmap_resized[binary_mask].mean())
            else:
                score = 0.0
 
            if gt_class_id in masks:
                masks[gt_class_id].append((mask_np, score))
            else:
                masks[gt_class_id] = [(mask_np, score)]
 
        batch_masks.append(masks)
 
    return batch_masks

# ------------------------------------------------------------------------------------------
# ------------------------------- GT Mask Embeddings (Experiment 2) ------------------------
# ------------------------------------------------------------------------------------------
#
# Reuses the exact same spatial-feature extraction `_predict_radio` above
# already exercises (via the SigLIP2 adaptor's `spatial_vision_features`),
# but skips the text-similarity step entirely -- Experiment 2 doesn't need
# text, just a visual embedding per mask.
#
# NOTE: RADIO builds often also expose the model's native, adaptor-
# independent backbone features under a `"backbone"` key in the same output
# dict (alongside each adaptor's key) -- that would arguably be a cleaner,
# fully text-agnostic embedding space than going through the SigLIP2
# adaptor's projection. I can't confirm that key exists in your installed
# build without seeing it, so this uses the adaptor path since that's
# already proven to work in your code. If you want to try the native
# backbone instead, swap the marked line below for
# `_, spatial_vision_features = vis_output["backbone"]` and confirm it runs.

def embed_gt_masks(model_name, model, images, batch_gt_masks, batch_gt_class_ids, device,
                    adaptor_name="siglip2", **kwargs):
    """
    Embed ground-truth instance masks using RADIO's dense spatial features,
    mean-pooled within each mask footprint and L2-normalized.
    **kwargs: accepted and ignored (e.g. gsam's `model_name` routing isn't
    needed here since this file only ever loads one model).
    """
    batch_results = []
    for image, gt_masks, gt_class_ids in zip(images, batch_gt_masks, batch_gt_class_ids):
        if len(gt_masks) == 0:
            batch_results.append([])
            continue

        image_tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor.div_(255.0)

        nearest_res = model.get_nearest_supported_resolution(*image_tensor.shape[-2:])
        image_tensor = F.interpolate(image_tensor, nearest_res, mode="bilinear", align_corners=False)
        feat_h = nearest_res[0] // model.patch_size
        feat_w = nearest_res[1] // model.patch_size

        with torch.no_grad():
            vis_output = model(image_tensor)

        # --- swap for vis_output["backbone"] if that key exists in your build ---
        _, spatial_vision_features = vis_output[adaptor_name]
        # --------------------------------------------------------------------------

        spatial_feats = spatial_vision_features.squeeze(0)  # (D, num_patches)
        spatial_feats = spatial_feats / spatial_feats.norm(dim=0, keepdim=True)
        feat_map = spatial_feats.reshape(-1, feat_h, feat_w).detach().cpu().numpy()  # (D, feat_h, feat_w)

        instances = []
        for mask, class_id in zip(gt_masks, gt_class_ids):
            mask_bool = mask.astype(bool)
            if mask_bool.sum() == 0:
                continue

            mask_u8 = (mask_bool.astype(np.uint8) * 255)
            mask_resized = cv2.resize(mask_u8, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST) > 0
            if not mask_resized.any():
                continue

            emb = feat_map[:, mask_resized].mean(axis=1)  # (D,)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            instances.append({"embedding": emb, "class_id": class_id})

        batch_results.append(instances)

    return batch_results


# ------------------------------------------------------------------------------------------
# ----------------------------- Helper API (Exposed Functions) -----------------------------
# ------------------------------------------------------------------------------------------

def init_model(model_name, prompt_class_names, device):
    """Loads and returns the C-RADIO model."""
    return _load_radio(prompt_class_names, device)

def predict_batch_masks(model, current_model_name, batch_images, batch_image_paths, prompt_class_ids, prompt_class_names, device):
    """Generates masks for a batch of images using C-RADIO."""
    return _predict_radio(model, batch_images, prompt_class_ids, prompt_class_names, device, threshold=0.6)

def cleanup_model(model):
    """Frees up GPU memory after all batches are done."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()