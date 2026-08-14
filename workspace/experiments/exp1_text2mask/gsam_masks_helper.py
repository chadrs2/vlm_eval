import copy
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from torchvision.ops import box_convert

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

import time
from metrics import (
    evaluate_batch,
    finalize_evaluation,
    print_evaluation_results,
)

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
        "save": False,#True,
    }
    model = SAM3SemanticPredictor(overrides=overrides)
    return model

def _load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)#'cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def load_gsam(all_class_names, device):
    # Load Grounding-DINO
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = _load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device=device)
    sam_checkpoint = '/workspace/models/sam_vit_b_01ec64.pth' # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return [groundingdino_model, sam_predictor]

def load_clip(all_class_names, device):
    return

def load_siglip(all_class_names, device):
    return

LOAD_MODEL_ENUM = {
    "yoloe": load_yoloe,
    "sam3": load_sam3,
    "gsam": load_gsam,
    "clip": load_clip,
    "siglip": load_siglip
}

def load_models(model_name, all_class_names, device="cuda:0"):
    available_models = ["yoloe", "sam3", "gsam", "clip", "siglip"]
    models = {}
    for am in available_models:
        if am == model_name or model_name == "all":
            models[am] = LOAD_MODEL_ENUM[am](all_class_names, device)
    return models

def predict_yoloe(model, images, gt_class_ids, device):
    
    results = model.predict(images)
    
    batch_masks = []
    
    for image, result in zip(images, results):
        IMG_W, IMG_H = image.shape[1], image.shape[0]
        
        masks = {}
        if result[0].masks is None:
            batch_masks.append(masks)
            continue
        
        yoloe_masks = result.masks.data
        yoloe_cls = result.boxes.cls
        for mask, cls in zip(yoloe_masks, yoloe_cls):
            class_idx = int(cls.item())  # Get class idx
            gt_class_id = gt_class_ids[class_idx]
            
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # Convert mask to 0-255
            mask_resized = cv2.resize(
                mask_np, 
                (IMG_W, IMG_H), 
                interpolation=cv2.INTER_NEAREST
            )

            if gt_class_id in masks:
                masks[gt_class_id].append(mask_resized)
            else:
                masks[gt_class_id] = [mask_resized]
        batch_masks.append(masks)
    
    return batch_masks

def predict_sam3(model, images, gt_class_ids, gt_class_names, device):
    batch_masks = []
    for image in images:
        IMG_W, IMG_H = image.shape[1], image.shape[0]
        
        masks = {}
        
        model.set_image(image)
        results = model(text=gt_class_names)
        sam3_masks = results[0].masks.data # num_found_prompts x H x W
        sam3_cls = results[0].boxes.cls
        for mask, cls in zip(sam3_masks, sam3_cls):
            class_idx = int(cls.item())  # Get class idx
            gt_class_id = gt_class_ids[class_idx]

            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # Convert mask to 0-255
            mask_resized = cv2.resize(
                mask_np, 
                (IMG_W, IMG_H), 
                interpolation=cv2.INTER_NEAREST
            )

            if gt_class_id in masks:
                masks[gt_class_id].append(mask_resized)
            else:
                masks[gt_class_id] = [mask_resized]
        batch_masks.append(masks)
        
    return batch_masks
        
def predict_gsam(model, image_paths, gt_class_ids, gt_class_names, device):
    TEXT_PROMPT = " . ".join(gt_class_names)
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    
    batch_masks = []
    for image_path in image_paths:
        image_source, image = load_image(image_path)
        IMG_H, IMG_W, _ = image_source.shape
        
        # G-DINO Box Predictions
        boxes, logits, phrases = predict(
            model=model[0], 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device=device
        )
        
        # SAM Mask Predictions
        model[1].set_image(image_source)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * \
            torch.Tensor([IMG_W, IMG_H, IMG_W, IMG_H])
        transformed_boxes = model[1].transform.apply_boxes_torch(
            boxes_xyxy, 
            image_source.shape[:2]
        ).to(device)
        gsam_masks, _, _ = model[1].predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        ) # num_masks x 1 x H x W

        masks = {}
        for mask, text in zip(gsam_masks, phrases):
            words = text.split(" ") if " " in text else [text]
            for st in words:
                if st in gt_class_names:
                    class_idx = gt_class_names.index(st)
                    gt_class_id = gt_class_ids[class_idx]
                    
                    mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)  # Convert mask to 0-255
                    mask_resized = cv2.resize(
                        mask_np, 
                        (IMG_W, IMG_H), 
                        interpolation=cv2.INTER_NEAREST
                    )
    
                    if gt_class_id in masks:
                        masks[gt_class_id].append(mask_resized)
                    else:
                        masks[gt_class_id] = [mask_resized]
        
        batch_masks.append(masks)
                
    return batch_masks


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
    void_class_ids=None
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

    models = load_models(model_name, eval_class_names, device)

    for model_name, model in models.items():
        print(f"Running {model_name}...")
        
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
            if model_name == "yoloe":
                batch_masks = predict_yoloe(
                    model,
                    batch_images,
                    eval_class_ids,
                    device
                )
            elif model_name == "sam3":
                batch_masks = predict_sam3(
                    model,
                    batch_images,
                    eval_class_ids,
                    eval_class_names,
                    device
                )
            elif model_name == "gsam":
                batch_masks = predict_gsam(
                    model,
                    batch_image_paths,
                    eval_class_ids,
                    eval_class_names,
                    device
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
                csv_path=os.path.join(output_folder,f"{model_name}_final_results.csv"),
            )
        else:
            results = finalize_evaluation(
                evaluation,
                eval_category_dict,
            )

        print_evaluation_results(results)