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
 
            if gt_class_id in masks:
                masks[gt_class_id].append(mask_np)
            else:
                masks[gt_class_id] = [mask_np]
 
        batch_masks.append(masks)
 
    return batch_masks

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