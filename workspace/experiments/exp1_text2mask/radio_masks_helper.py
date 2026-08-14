import os
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
import cv2
from metrics import (
    evaluate_batch,
    finalize_evaluation,
    print_evaluation_results,
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_radio(
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
    # model = model.cuda().eval()
    model = model.to(device=device).eval()
    return model
 
 
# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_radio(
    model,
    images,
    gt_class_ids,
    gt_class_names,
    device,
    adaptor_name="siglip2",
    threshold=0.6,
):
    """
    Runs C-RADIO + SigLIP2 adaptor over a batch of images and produces a
    thresholded cosine-similarity heatmap mask per text prompt, following
    the same batch_masks dict-of-lists structure used by the other
    predict_* functions (gt_class_id -> [mask_np, ...]).
    """
    adaptor = model.adaptors[adaptor_name]
    text_input = adaptor.tokenizer(gt_class_names).to(device)
    with torch.no_grad():
        text_tokens = adaptor.encode_text(text_input, normalize=True)  # num_classes x D
 
    batch_masks = []
    for image in images:
        IMG_H, IMG_W = image.shape[0], image.shape[1]
 
        # numpy HWC uint8 image -> normalized CHW tensor on device
        image_tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor.div_(255.0)  # RADIO expects values between 0 and 1
 
        nearest_res = model.get_nearest_supported_resolution(*image_tensor.shape[-2:])
        image_tensor = F.interpolate(
            image_tensor, nearest_res, mode="bilinear", align_corners=False
        )
        feat_h = nearest_res[0] // model.patch_size
        feat_w = nearest_res[1] // model.patch_size
 
        with torch.no_grad():
            vis_output = model(image_tensor)
        s2_sum, spatial_vision_features = vis_output[adaptor_name]
 
        spatial_feats = spatial_vision_features.squeeze(0)  # D x H*W
        spatial_feats = spatial_feats / spatial_feats.norm(dim=0, keepdim=True)
 
        # text_tokens: num_classes x D, spatial_feats_flat: D x (H*W)
        similarity = torch.matmul(text_tokens, spatial_feats.T)  # num_classes x (H*W)
 
        masks = {}
        for c_idx, class_name in enumerate(gt_class_names):
            gt_class_id = gt_class_ids[c_idx]
 
            heatmap = similarity[c_idx].view(feat_h, feat_w).detach().cpu().numpy()
            heatmap_resized = cv2.resize(
                heatmap,
                (IMG_W, IMG_H),
                interpolation=cv2.INTER_LINEAR,
            )
 
            # Normalize heatmap values cleanly between 0 and 1
            h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
            heatmap_resized = (heatmap_resized - h_min) / (h_max - h_min + 1e-8)
 
            binary_mask = heatmap_resized > threshold
            mask_np = (binary_mask.astype(np.uint8) * 255)  # Convert mask to 0-255
 
            if gt_class_id in masks:
                masks[gt_class_id].append(mask_np)
            else:
                masks[gt_class_id] = [mask_np]
 
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
    threshold=0.6,
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
 
 
    model = load_radio(eval_class_names, device)

    print(f"Running radio...")
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
        batch_masks = predict_radio(
            model,
            batch_images,
            eval_class_ids,
            eval_class_names,
            device,
            threshold=threshold,
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
 