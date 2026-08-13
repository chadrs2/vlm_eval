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
# segment anything
from segment_anything.segment_anything import build_sam, SamPredictor, build_sam_vit_b 

import time
from metrics import (
    evaluate_batch,
    finalize_evaluation,
    print_evaluation_results,
)

def load_yoloe(all_class_names):
    model = YOLOE("yoloe-26m-seg.pt")
    model.set_classes(all_class_names)
    return model

def load_sam3(all_class_names):
    return

def load_gsam(all_class_names):
    return

def load_clip(all_class_names):
    return

def load_siglip(all_class_names):
    return

LOAD_MODEL_ENUM = {
    "yoloe": load_yoloe,
    "sam3": load_sam3,
    "gsam": load_gsam,
    "clip": load_clip,
    "siglip": load_siglip
}

def load_models(model_name, all_class_names):
    available_models = ["yoloe", "sam3", "gsam", "clip", "siglip"]
    models = {}
    for am in available_models:
        if am == model_name or model_name == "all":
            models[am] = LOAD_MODEL_ENUM[am](all_class_names)
    return models

def predict_yoloe(model, images, gt_class_ids):
    
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
            class_idx = int(cls.item())  # Get class id
            gt_class_id = gt_class_ids[class_idx]
            
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # Convert mask to 0-255
            mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

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
    batch_size=8
):
    all_class_ids = [k for k, v in category_name_dict.items()]
    all_class_names = [v for k, v in category_name_dict.items()]
    models = load_models(model_name, all_class_names)
    
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
                    all_class_ids,
                )
            runtime = time.perf_counter() - start_time
            
            # TODO: Evaluate batch masks and save results
            evaluation = evaluate_batch(
                batch_masks=batch_masks,
                batch_gt_masks=batch_gt_masks,
                batch_class_ids=batch_class_ids,
                batch_images=batch_images,
                class_ids=all_class_ids,
                accumulator=evaluation,
                runtime=runtime,
            )
    
        # ---------------------------------------------------------
        # Finalize metrics over entire dataset
        # ---------------------------------------------------------
        if output_folder:
            results = finalize_evaluation(
                evaluation,
                category_name_dict,
                csv_path=os.path.join(output_folder,f"{model_name}_final_results.csv"),
            )
        else:
            results = finalize_evaluation(
                evaluation,
                category_name_dict,
            )

        print_evaluation_results(results)
            