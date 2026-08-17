import copy
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from torchvision.ops import box_convert
import gc

from transformers import AutoProcessor, AutoModel
from ultralytics import settings
settings.update({"weights_dir": "/workspace/models"})
from ultralytics import YOLOE, FastSAM
from ultralytics.nn.text_model import CLIP
from ultralytics.models.sam import SAM3SemanticPredictor
import supervision as sv

import os, sys
sys.path.append("/workspace/projects/Grounded-Segment-Anything")
sys.path.append("/workspace/projects/Grounded-Segment-Anything/GroundingDINO")

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download
# segment anything
from segment_anything.segment_anything import build_sam, SamPredictor, build_sam_vit_b 

# ------------------------------------------------------------------------------------------
# -------------------------------------- Load Models ---------------------------------------
# ------------------------------------------------------------------------------------------

def load_yoloe(all_class_names, device):
    model = YOLOE("yoloe-26m-seg.pt").to(device=device)
    model.set_classes(all_class_names)
    return model

def load_sam3(all_class_names, device):
    overrides = {
        "conf": 0.25,
        "task": "segment",
        "mode": "predict",
        "model": "/workspace/models/sam3.pt",
        "quantize": 16,  # Use FP16 for faster inference
        "device": device,
        "save": False,
    }
    model = SAM3SemanticPredictor(overrides=overrides)
    return model

def _load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def load_gsam(all_class_names, device):
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    
    ckpt_filename = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py" 

    # swinB weights
    # ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    # ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = _load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device=device)
    
    sam_checkpoint = '/workspace/models/sam_vit_b_01ec64.pth' 
    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return [groundingdino_model, sam_predictor]

def load_clip(all_class_names, device):
    clip_model = CLIP(size="ViT-B/32", device=device)
    fastsam_model = FastSAM(model="FastSAM-s.pt")
    return [fastsam_model, clip_model]

def load_siglip(all_class_names, device):
    fastsam_model = FastSAM(model="FastSAM-s.pt")
    model_id = "google/siglip-base-patch16-224"
    siglip_processor = AutoProcessor.from_pretrained(model_id)
    siglip_model = AutoModel.from_pretrained(model_id).to(device)
    return [fastsam_model, siglip_processor, siglip_model]

LOAD_MODEL_ENUM = {
    "yoloe": load_yoloe,
    "sam3": load_sam3,
    "gsam": load_gsam,
    "clip": load_clip,
    "siglip": load_siglip
}

# ------------------------------------------------------------------------------------------
# ----------------------------------- Model Predictions ------------------------------------
# ------------------------------------------------------------------------------------------

def predict_yoloe(model, images, prompt_class_ids, device):
    results = model.predict(images, conf=0.1)
    batch_masks = []
    
    for image, result in zip(images, results):
        IMG_W, IMG_H = image.shape[1], image.shape[0]
        masks = {}
        if result.masks is None:
            batch_masks.append(masks)
            continue
        
        yoloe_masks = result.masks.data
        yoloe_cls = result.boxes.cls
        for mask, cls in zip(yoloe_masks, yoloe_cls):
            class_idx = int(cls.item())
            gt_class_id = prompt_class_ids[class_idx]
            
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            if gt_class_id in masks:
                masks[gt_class_id].append(mask_resized)
            else:
                masks[gt_class_id] = [mask_resized]
        batch_masks.append(masks)
    
    return batch_masks

def predict_sam3(model, images, prompt_class_ids, prompt_class_names, device):
    batch_masks = []
    for image in images:
        IMG_W, IMG_H = image.shape[1], image.shape[0]
        masks = {}
        
        model.set_image(image)
        results = model(text=prompt_class_names)
        if results[0].masks is None:
            batch_masks.append(masks)
            continue

        sam3_masks = results[0].masks.data 
        sam3_cls = results[0].boxes.cls
        for mask, cls in zip(sam3_masks, sam3_cls):
            class_idx = int(cls.item())  
            gt_class_id = prompt_class_ids[class_idx]

            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            if gt_class_id in masks:
                masks[gt_class_id].append(mask_resized)
            else:
                masks[gt_class_id] = [mask_resized]
        batch_masks.append(masks)
        
    return batch_masks

def predict_gsam(model, image_paths, prompt_class_ids, prompt_class_names, device):
    TEXT_PROMPT = " . ".join(prompt_class_names)
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    
    batch_masks = []
    for image_path in image_paths:
        image_source, image = load_image(image_path)
        IMG_H, IMG_W, _ = image_source.shape
        
        boxes, logits, phrases = predict(
            model=model[0], image=image, caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD, device=device
        )
        
        model[1].set_image(image_source)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([IMG_W, IMG_H, IMG_W, IMG_H])
        transformed_boxes = model[1].transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        gsam_masks, _, _ = model[1].predict_torch(
            point_coords = None, point_labels = None, boxes = transformed_boxes, multimask_output = False,
        ) 

        masks = {}
        for mask, text in zip(gsam_masks, phrases):
            # Iterate through the custom prompts directly instead of splitting by space
            for class_name in prompt_class_names:
                # Substring matching handles Grounding DINO's tendency to slightly alter phrases
                if class_name.lower() in text.lower() or text.lower() in class_name.lower():
                    class_idx = prompt_class_names.index(class_name)
                    gt_class_id = prompt_class_ids[class_idx]
                    
                    mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8) 
                    mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    
                    if gt_class_id in masks:
                        masks[gt_class_id].append(mask_resized)
                    else:
                        masks[gt_class_id] = [mask_resized]
        batch_masks.append(masks)
                
    return batch_masks

def _get_fsam_masks(fsam_model, images, device):
    all_img_masks = []
    all_fsam_masks = []
    IMG_W, IMG_H = images[0].shape[1], images[0].shape[0]
    fsam_results = fsam_model(
        images, device=device, retina_masks=True, imgsz=(IMG_H, IMG_W), conf=0.003, iou=0.25, max_det=100,
    )
    
    buffer = 10
    kernel = np.ones((5, 5), np.uint8)
    for image, result in zip(images, fsam_results):
        if result.masks is None:
            all_fsam_masks.append(torch.empty(0, dtype=torch.bool))
            all_img_masks.append([])
            continue

        fsam_masks = result.masks.data.bool()
        all_fsam_masks.append(fsam_masks)
        
        img_masks = []
        for mask in fsam_masks:
            mask_bool = (mask.cpu().numpy() > 0)
            if mask_bool.sum() == 0:
                continue
    
            mask_u8 = (mask_bool.astype(np.uint8) * 255)
            x, y, w, h = cv2.boundingRect(mask_u8)
            x1, y1 = max(0, x - buffer), max(0, y - buffer)
            x2, y2 = min(IMG_W, x + w + buffer), min(IMG_H, y + h + buffer)
    
            dilated = cv2.dilate(mask_u8, kernel, iterations=3).astype(bool)
            if not dilated.any():
                continue
    
            cropped_bb = image[y1:y2, x1:x2].copy()
            roi = dilated[y1:y2, x1:x2]
            cropped_bb[~roi] = 0
            try:
                pil_crop = Image.fromarray(cropped_bb)
            except Exception:
                pil_crop = Image.fromarray((np.clip(cropped_bb, 0, 255)).astype(np.uint8))
    
            img_masks.append(pil_crop)
        all_img_masks.append(img_masks)
    
    return all_img_masks, all_fsam_masks

def _get_clip_mask_embs(fsam_model, clip_model, images, device):
    clip_mask_embs = []
    all_img_masks, all_fsam_masks = _get_fsam_masks(fsam_model, images, device)
    
    for img_masks in all_img_masks:
        if len(img_masks) == 0:
            clip_mask_embs.append(None)
            continue

        preprocessed = [clip_model.image_preprocess(img).unsqueeze(0) for img in img_masks]
        clip_input_batch = torch.cat(preprocessed, dim=0).to(device, non_blocking=True)

        with torch.no_grad():
            fsam_clip_embs_tensor = clip_model.encode_image(clip_input_batch)
        clip_mask_embs.append(fsam_clip_embs_tensor)
    return clip_mask_embs, all_fsam_masks

def predict_clip(model, images, prompt_class_ids, prompt_class_names, device):
    clip_mask_embs, all_fsam_masks = _get_clip_mask_embs(model[0], model[1], images, device)
    clip_prompts = [f"a photo of a {class_name}" for class_name in prompt_class_names]
    toks = model[1].tokenize(clip_prompts)
    clip_txt_embs = model[1].encode_text(toks)
    
    COSSIM_THRESHOLD = 0.25
    batch_masks = []
    for img_idx in range(len(images)):
        IMG_W, IMG_H = images[img_idx].shape[1], images[img_idx].shape[0]
        merged_masks = {}
        if clip_mask_embs[img_idx] is None:
            batch_masks.append(merged_masks)
            continue

        similarity = F.cosine_similarity(clip_mask_embs[img_idx].unsqueeze(1), clip_txt_embs.unsqueeze(0), dim=-1)

        for c_idx, class_name in enumerate(prompt_class_names):
            gt_class_id = prompt_class_ids[c_idx]
            class_sims = similarity[:, c_idx]
            passing_indices = (class_sims > COSSIM_THRESHOLD).nonzero(as_tuple=True)[0]
                
            if len(passing_indices) == 0:
                continue
        
            selected_masks = all_fsam_masks[img_idx][passing_indices]
            combined_mask = selected_masks.any(dim=0)  
            mask_np = (combined_mask.squeeze().cpu().numpy() * 255).astype(np.uint8) 
            mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            if gt_class_id in merged_masks:
                merged_masks[gt_class_id].append(mask_resized)
            else:
                merged_masks[gt_class_id] = [mask_resized]
        batch_masks.append(merged_masks)
        
    return batch_masks

def _get_siglip_mask_embs(fsam_model, siglip_processor, siglip_model, images, device):
    siglip_mask_embs = []
    all_img_masks, all_fsam_masks = _get_fsam_masks(fsam_model, images, device)
    
    for img_masks in all_img_masks:
        if len(img_masks) == 0:
            siglip_mask_embs.append(None)
            continue

        image_inputs = siglip_processor(images=img_masks, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            image_embeds = siglip_model.get_image_features(**image_inputs)
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        siglip_mask_embs.append(image_embeds)
    
    return siglip_mask_embs, all_fsam_masks

def predict_siglip(model, images, prompt_class_ids, prompt_class_names, device):
    siglip_mask_embs, all_fsam_masks = _get_siglip_mask_embs(model[0], model[1], model[2], images, device)
    
    siglip_prompts = [f"a photo of a {class_name}" for class_name in prompt_class_names]
    text_inputs = model[1](text=siglip_prompts, return_tensors="pt", padding="max_length").to(device)
    with torch.no_grad():
        siglip_text_embeds = model[2].get_text_features(**text_inputs)
        siglip_text_embeds = F.normalize(siglip_text_embeds, p=2, dim=-1)
    
    COSSIM_THRESHOLD = 0.06
    batch_masks = []
    for img_idx in range(len(images)):
        IMG_W, IMG_H = images[img_idx].shape[1], images[img_idx].shape[0]
        merged_masks = {}
        if siglip_mask_embs[img_idx] is None:
            batch_masks.append(merged_masks)
            continue

        similarity = F.cosine_similarity(siglip_mask_embs[img_idx].unsqueeze(1), siglip_text_embeds.unsqueeze(0), dim=-1)

        for c_idx, class_name in enumerate(prompt_class_names):
            gt_class_id = prompt_class_ids[c_idx]
            class_sims = similarity[:, c_idx]
            passing_indices = (class_sims > COSSIM_THRESHOLD).nonzero(as_tuple=True)[0]
                
            if len(passing_indices) == 0:
                continue
        
            selected_masks = all_fsam_masks[img_idx][passing_indices]
            combined_mask = selected_masks.any(dim=0)
            mask_np = (combined_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            if gt_class_id in merged_masks:
                merged_masks[gt_class_id].append(mask_resized)
            else:
                merged_masks[gt_class_id] = [mask_resized]
        batch_masks.append(merged_masks)
        
    return batch_masks

# ------------------------------------------------------------------------------------------
# ----------------------------- Helper API (Exposed Functions) -----------------------------
# ------------------------------------------------------------------------------------------

def init_model(model_name, prompt_class_names, device):
    """Loads and returns the specified model."""
    if model_name not in LOAD_MODEL_ENUM:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(LOAD_MODEL_ENUM.keys())}.")
    return LOAD_MODEL_ENUM[model_name](prompt_class_names, device)

def predict_batch_masks(model, current_model_name, batch_images, batch_image_paths, prompt_class_ids, prompt_class_names, device):
    """Routes the batch to the correct prediction function based on model name."""
    if current_model_name == "yoloe":
        return predict_yoloe(model, batch_images, prompt_class_ids, device)
    elif current_model_name == "sam3":
        return predict_sam3(model, batch_images, prompt_class_ids, prompt_class_names, device)
    elif current_model_name == "gsam":
        return predict_gsam(model, batch_image_paths, prompt_class_ids, prompt_class_names, device)
    elif current_model_name == "clip":
        return predict_clip(model, batch_images, prompt_class_ids, prompt_class_names, device)
    elif current_model_name == "siglip":
        return predict_siglip(model, batch_images, prompt_class_ids, prompt_class_names, device)
    else:
        raise ValueError(f"Unknown model name: {current_model_name}")

def cleanup_model(model):
    """Frees up GPU memory after all batches are done."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()