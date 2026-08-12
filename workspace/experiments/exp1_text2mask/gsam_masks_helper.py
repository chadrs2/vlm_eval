import copy
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from torchvision.ops import box_convert

from transformers import AutoProcessor, AutoModel
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

def load_yoloe():
    return YOLOE("/workspace/models/yoloe-26m-seg.pt")

def load_sam3():
    return

def load_gsam():
    return

def load_clip():
    return

def load_siglip():
    return

LOAD_MODEL_ENUM = {
    "yoloe": load_yoloe,
    "sam3": load_sam3,
    "gsam": load_gsam,
    "clip": load_clip,
    "siglip": load_siglip
}

def load_models(model_name):
    available_models = ["yoloe", "sam3", "gsam", "clip", "siglip"]
    models = {}
    for am in available_models:
        if am == model_name or model_name == "all":
            models[am] = LOAD_MODEL_ENUM[am]
    return models

def predict_yoloe(model, image, class_names):
    masks = {}
    
    model.set_classes(class_names)
    results = model.predict(image)
    yoloe_masks = results[0].masks.data
    for cls_id in range(len(class_names)):
        if yoloe_masks[cls_id].sum() == 0:
            continue
        masks[cls_id] = yoloe_masks[cls_id]
    
    return masks
    

def run_experiment1(images, gt_masks, class_ids, class_names, model_name, output_folder):
    models = load_models(model_name)
    
    for model_name, model in  models:
        
        # TODO: Make a batch process
        for img_idx in range(len(images)):
            curr_class_names = class_names[img_idx]
            unique_curr_class_names = np.unique(curr_class_names)
        