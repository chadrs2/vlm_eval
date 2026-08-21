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

# raise specific torch flags to optimize CLIP performance
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        yoloe_conf = result.boxes.conf
        for mask, cls, conf in zip(yoloe_masks, yoloe_cls, yoloe_conf):
            class_idx = int(cls.item())
            gt_class_id = prompt_class_ids[class_idx]
            score = float(conf.item())
            
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            if gt_class_id in masks:
                masks[gt_class_id].append((mask_resized, score))
            else:
                masks[gt_class_id] = [(mask_resized, score)]
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
        sam3_conf = results[0].boxes.conf
        for mask, cls, conf in zip(sam3_masks, sam3_cls, sam3_conf):
            class_idx = int(cls.item())  
            gt_class_id = prompt_class_ids[class_idx]
            score = float(conf.item())

            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            if gt_class_id in masks:
                masks[gt_class_id].append((mask_resized, score))
            else:
                masks[gt_class_id] = [(mask_resized, score)]
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
        
        # ---------------------------------------------------------
        # NEW SAFETY CHECK: If no boxes are found, skip SAM entirely
        # ---------------------------------------------------------
        if boxes.shape[0] == 0:
            batch_masks.append({})
            continue
        # ---------------------------------------------------------
        
        model[1].set_image(image_source)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([IMG_W, IMG_H, IMG_W, IMG_H])
        transformed_boxes = model[1].transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        gsam_masks, _, _ = model[1].predict_torch(
            point_coords = None, point_labels = None, boxes = transformed_boxes, multimask_output = False,
        ) 

        masks = {}
        # `logits` holds GroundingDINO's per-box confidence...
        for mask, text, logit in zip(gsam_masks, phrases, logits):
            score = float(logit.item())
            # Iterate through the custom prompts directly instead of splitting by space
            for class_name in prompt_class_names:
                # Substring matching handles Grounding DINO's tendency to slightly alter phrases
                if class_name.lower() in text.lower() or text.lower() in class_name.lower():
                    class_idx = prompt_class_names.index(class_name)
                    gt_class_id = prompt_class_ids[class_idx]
                    
                    mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8) 
                    mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    
                    if gt_class_id in masks:
                        masks[gt_class_id].append((mask_resized, score))
                    else:
                        masks[gt_class_id] = [(mask_resized, score)]
        batch_masks.append(masks)
                
    return batch_masks

def _get_fsam_masks(fsam_model, images, device):
    """
    Run FastSAM and build per-instance crops for a batch of images.

    NOTE: images in a batch can come from different source datasets (or
    just vary in size/aspect ratio within one dataset -- COCO, LaRS, and
    GOOSE all have mixed resolutions), so we run FastSAM one image at a
    time instead of handing the whole batch to `fsam_model` in one call.
    Passing a mixed-resolution list through with a single `imgsz` forces
    every image but the first onto that first image's canvas, distorting
    (stretching/squashing) their aspect ratio before segmentation. Looping
    keeps each image at its own native resolution -- true full-resolution,
    fair, apples-to-apples inference across datasets.
    """
    all_img_masks = []
    all_fsam_masks = []

    buffer = 10
    kernel = np.ones((5, 5), np.uint8)
    for image in images:
        IMG_W, IMG_H = image.shape[1], image.shape[0]
        result = fsam_model(
            [image], device=device, retina_masks=True, imgsz=(IMG_H, IMG_W), conf=0.003, iou=0.25, max_det=100,
        )[0]

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

            for idx in passing_indices:
                mask = all_fsam_masks[img_idx][idx]
                score = float(class_sims[idx].item())

                mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

                if gt_class_id in merged_masks:
                    merged_masks[gt_class_id].append((mask_resized, score))
                else:
                    merged_masks[gt_class_id] = [(mask_resized, score)]
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

            # Same un-merging as predict_clip above -- keep each passing
            # FastSAM proposal as its own (mask, score) instance.
            for idx in passing_indices:
                mask = all_fsam_masks[img_idx][idx]
                score = float(class_sims[idx].item())

                mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

                if gt_class_id in merged_masks:
                    merged_masks[gt_class_id].append((mask_resized, score))
                else:
                    merged_masks[gt_class_id] = [(mask_resized, score)]
        batch_masks.append(merged_masks)
        
    return batch_masks

# ------------------------------------------------------------------------------------------
# ------------------------------- GT Mask Embeddings (Experiment 2) ------------------------
# ------------------------------------------------------------------------------------------
#
# "clip" / "siglip": crop each GT mask (bbox + buffer, background zeroed
# outside a dilated mask -- same convention as _get_fsam_masks above) and
# run it through the encoder as a standalone image.
#
# "gsam" / "sam3": these encoders (GroundingDINO's Swin backbone, and
# SAM3's Perception-Encoder backbone) are trained to produce DENSE
# per-pixel/patch features for a whole natural-image, not a single pooled
# vector for a small standalone crop the way CLIP/SigLIP are. Feeding them
# a cropped, mostly-blacked-out mask region (like the clip/siglip path
# does) pushes them out of their training distribution. Instead we run
# the encoder ONCE on the full original image and do masked-average-pooling
# over the dense feature map restricted to each GT mask's footprint (mask
# resized down to the encoder's native feature-grid resolution). This is
# the standard way these dense encoders are turned into per-object
# embeddings.
#
# "gsam" specifically uses GroundingDINO's Swin backbone rather than SAM's
# ViT-B decoder-side encoder: SAM's decoder never sees class identity (it
# just turns a box into a mask), whereas GroundingDINO's backbone is the
# part of GSAM actually doing recognition, and pulled here BEFORE the
# feature-enhancer's text cross-attention fusion so it stays independent
# of whatever class prompt happens to be active.
#
# "yoloe" is still NOT supported: Ultralytics doesn't expose a documented
# embedding API for it, and guessing at undocumented internal attributes
# would risk silently-wrong embeddings rather than a clean error.
#
# NOTE on SAM3: `SAM3SemanticPredictor.features` (set by `.set_image()`) is
# used in Ultralytics' own docs to reuse features across queries via
# `inference_features(predictor.features, ...)`, but its exact tensor
# shape isn't documented publicly -- only the usage pattern is. Before
# trusting `_sam3_embed_gt_masks` below, run `_inspect_sam3_features()`
# once in your environment to confirm the shape assumptions hold for your
# installed ultralytics version.

def _record_skip(skip_log, image_id, class_id, reason):
    """
    Append a record of a GT instance that could not be embedded. `skip_log`
    is an optional list passed in by the caller (run_experiment2.py); if
    None, skips are silently dropped as before (backward compatible).
    Every *_embed_gt_masks path below calls this instead of a bare
    `continue`, so instance loss is visible and attributable (which class,
    which image, why) rather than just showing up as a lower final count.
    """
    if skip_log is not None:
        skip_log.append({"image_id": image_id, "class_id": class_id, "reason": reason})


def _crop_mask_region(image, mask, buffer=10, kernel=None):
    """Bounding-box crop of `mask` in `image`, background zeroed outside a
    dilated version of the mask. Returns None if the mask is empty."""
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)

    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return None

    IMG_H, IMG_W = image.shape[0], image.shape[1]
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    x, y, w, h = cv2.boundingRect(mask_u8)
    x1, y1 = max(0, x - buffer), max(0, y - buffer)
    x2, y2 = min(IMG_W, x + w + buffer), min(IMG_H, y + h + buffer)

    dilated = cv2.dilate(mask_u8, kernel, iterations=3).astype(bool)
    if not dilated.any():
        return None

    cropped_bb = image[y1:y2, x1:x2].copy()
    roi = dilated[y1:y2, x1:x2]
    cropped_bb[~roi] = 0

    try:
        return Image.fromarray(cropped_bb)
    except Exception:
        return Image.fromarray(np.clip(cropped_bb, 0, 255).astype(np.uint8))


def _clip_embed_crops(clip_model, pil_imgs, device):
    preprocessed = [clip_model.image_preprocess(img).unsqueeze(0) for img in pil_imgs]
    batch = torch.cat(preprocessed, dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        embs = clip_model.encode_image(batch)
        embs = F.normalize(embs, p=2, dim=-1)
    return embs


def _siglip_embed_crops(siglip_processor, siglip_model, pil_imgs, device):
    inputs = siglip_processor(images=pil_imgs, return_tensors="pt", padding="max_length").to(device)
    with torch.no_grad():
        embs = siglip_model.get_image_features(**inputs)
        embs = F.normalize(embs, p=2, dim=-1)
    return embs


def _embed_crop_based(embed_fn, images, batch_gt_masks, batch_gt_class_ids,
                       batch_image_ids=None, skip_log=None):
    """Shared crop-and-encode loop for clip/siglip."""
    if batch_image_ids is None:
        batch_image_ids = [None] * len(images)

    batch_results = []
    for image, gt_masks, gt_class_ids, image_id in zip(images, batch_gt_masks, batch_gt_class_ids, batch_image_ids):
        crops, kept_class_ids = [], []
        for mask, class_id in zip(gt_masks, gt_class_ids):
            crop = _crop_mask_region(image, mask)
            if crop is None:
                _record_skip(skip_log, image_id, class_id, "empty/degenerate GT mask (no crop region)")
                continue
            crops.append(crop)
            kept_class_ids.append(class_id)

        if len(crops) == 0:
            batch_results.append([])
            continue

        embeddings = embed_fn(crops).detach().cpu().numpy()
        batch_results.append([
            {"embedding": emb, "class_id": cid}
            for emb, cid in zip(embeddings, kept_class_ids)
        ])

    return batch_results


def _mask_pool_embedding(feat_map, mask):
    """
    Masked-average-pool a dense (C, H', W') feature map down to a single
    (C,) vector, restricted to the region a full-resolution binary `mask`
    (H, W) covers. The mask is nearest-neighbor-resized down to the
    feature grid; if that collapses to nothing (mask smaller than one
    feature cell), we fall back to whichever cell(s) the mask overlaps at
    all via a >0 threshold on a linearly-resized mask.
    """
    C, Hf, Wf = feat_map.shape
    mask_u8 = mask.astype(np.uint8)

    mask_small = cv2.resize(mask_u8, (Wf, Hf), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not mask_small.any():
        mask_small = cv2.resize(mask_u8, (Wf, Hf), interpolation=cv2.INTER_LINEAR) > 0
    if not mask_small.any():
        # Degenerate: mask has no area at all even under this fallback.
        return None

    flat = feat_map.reshape(C, -1)
    idx = torch.from_numpy(mask_small.reshape(-1)).to(feat_map.device)
    return flat[:, idx].mean(dim=1)


# GroundingDINO's own transform pipeline (mirrors GroundingDINO.util.inference.load_image,
# which this file's predict_gsam relies on indirectly via load_image(image_path)).
# Reproduced here because embed_gt_masks receives decoded np.ndarray images, not paths.
_GDINO_TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _groundingdino_embed_gt_masks(groundingdino_model, images, batch_gt_masks, batch_gt_class_ids, device,
                                   batch_image_ids=None, skip_log=None):
    """
    Embed GT masks using GroundingDINO's own Swin-Transformer visual
    backbone -- the part of GSAM that actually recognizes objects (SAM's
    decoder just turns a box into a mask; it never sees class identity).

    `groundingdino_model.backbone` is a Joiner(Swin, position_encoding),
    the standard DETR/Deformable-DETR pattern this repo inherits from.
    Called directly like this, it returns RAW multi-scale Swin feature
    maps -- BEFORE the feature-enhancer's text cross-attention fusion, so
    the resulting embedding is not conditioned on whatever class prompt
    happens to be active. Each scale is masked-average-pooled separately
    and concatenated into one vector per instance.

    Verify `groundingdino_model.backbone` exists in your installed
    checkout before trusting this (same "don't guess at undocumented
    internals" caveat as with SAM3) -- e.g.:
        print(type(groundingdino_model.backbone))
    """
    from GroundingDINO.groundingdino.util.misc import nested_tensor_from_tensor_list

    if batch_image_ids is None:
        batch_image_ids = [None] * len(images)

    batch_results = []
    for image, gt_masks, gt_class_ids, image_id in zip(images, batch_gt_masks, batch_gt_class_ids, batch_image_ids):
        if len(gt_masks) == 0:
            batch_results.append([])
            continue

        pil_image = Image.fromarray(image).convert("RGB")
        image_transformed, _ = _GDINO_TRANSFORM(pil_image, None)
        samples = nested_tensor_from_tensor_list([image_transformed]).to(device)

        with torch.no_grad():
            gdino_features, _poss = groundingdino_model.backbone(samples)
        # gdino_features: list of NestedTensor, one per Swin stage, each
        # .tensors shaped (1, C_l, H_l, W_l) at that stage's stride.
        feat_maps = [f.tensors.squeeze(0) for f in gdino_features]  # list of (C_l, H_l, W_l)

        result = []
        for mask, class_id in zip(gt_masks, gt_class_ids):
            if mask.astype(bool).sum() == 0:
                _record_skip(skip_log, image_id, class_id, "empty GT mask (0 pixels)")
                continue
            level_embs = [_mask_pool_embedding(fm, mask) for fm in feat_maps]
            n_collapsed = sum(1 for e in level_embs if e is None)
            if n_collapsed > 0:
                _record_skip(
                    skip_log, image_id, class_id,
                    f"mask collapsed to 0 feature cells at {n_collapsed}/{len(feat_maps)} "
                    f"Swin backbone scales (object too small relative to that scale's stride)",
                )
                continue
            emb = torch.cat(level_embs, dim=0)  # concat across Swin stages
            emb = F.normalize(emb, p=2, dim=-1).detach().cpu().numpy()
            result.append({"embedding": emb, "class_id": class_id})
        batch_results.append(result)
    return batch_results


def _inspect_sam3_features(sam3_predictor, image):
    """
    One-off debugging helper -- run this in your environment BEFORE
    trusting `_sam3_embed_gt_masks`. Prints the type/shape of
    `predictor.features` so you can confirm (or fix) the extraction
    assumptions made below, per this file's policy of not guessing at
    undocumented internal attributes.
    """
    sam3_predictor.set_image(image)
    feats = sam3_predictor.features
    print(f"type(predictor.features) = {type(feats)}")
    if isinstance(feats, (list, tuple)):
        for i, f in enumerate(feats):
            print(f"  [{i}] type={type(f)}", getattr(f, "shape", "(no .shape)"))
    elif isinstance(feats, dict):
        for k, v in feats.items():
            print(f"  ['{k}'] type={type(v)}", getattr(v, "shape", "(no .shape)"))
    elif hasattr(feats, "shape"):
        print(f"  shape={feats.shape}")
    else:
        print(f"  repr={feats!r}")
    return feats


def _sam3_extract_feat_map(feats, expect_channels_last=False):
    """
    Best-effort extraction of a single (C, H, W) dense feature map out of
    `predictor.features`. Handles the common cases: a bare tensor, a
    list/tuple of multi-level feature tensors (we take the last/highest
    level, matching `num_feature_levels=1` default usage), or a dict with
    an obvious image-feature key. VERIFY against `_inspect_sam3_features`
    output for your installed ultralytics version -- adjust this function
    if the structure doesn't match.
    """
    f = feats
    if isinstance(f, dict):
        # Confirmed via _inspect_sam3_features() on the installed ultralytics
        # version: predictor.features = {'vision_features': (1, 256, 46, 46),
        # 'vision_pos_enc': [...], 'backbone_fpn': [...], 'sam2_backbone_out': {...}}.
        # 'vision_features' is the dense (B, C, H, W) map we want; the others
        # are positional encodings / raw FPN levels / a nested backbone dict,
        # none of which are a single ready-to-pool feature tensor.
        for key in ("vision_features", "vision_feats", "image_features", "feats", "src"):
            if key in f:
                f = f[key]
                break
    if isinstance(f, (list, tuple)):
        f = f[-1]

    if f.dim() == 4:      # (B, C, H, W) or (B, H, W, C)
        f = f.squeeze(0)
    if f.dim() == 3 and expect_channels_last:  # (H, W, C) -> (C, H, W)
        f = f.permute(2, 0, 1)
    elif f.dim() == 2:    # (HW, C) flattened, channels-last -> need H, W externally
        raise ValueError(
            "predictor.features flattened to (HW, C) with no spatial shape "
            "recoverable here -- pass vis_feat_sizes/spatial_shapes from "
            "your ultralytics version to reshape before pooling."
        )
    return f  # (C, H, W)


def _sam3_embed_gt_masks(sam3_predictor, images, batch_gt_masks, batch_gt_class_ids,
                          batch_image_ids=None, skip_log=None):
    """
    Embed GT masks using SAM3's own Perception-Encoder vision backbone.
    Mirrors the other embed_gt_masks branches: one feature pull per full
    image, then masked-average-pooling per GT instance.

    Run `_inspect_sam3_features()` first to confirm `_sam3_extract_feat_map`
    matches your installed ultralytics/SAM3 version's `.features` layout.
    """
    if batch_image_ids is None:
        batch_image_ids = [None] * len(images)

    batch_results = []
    for image, gt_masks, gt_class_ids, image_id in zip(images, batch_gt_masks, batch_gt_class_ids, batch_image_ids):
        if len(gt_masks) == 0:
            batch_results.append([])
            continue

        sam3_predictor.set_image(image)
        feat_map = _sam3_extract_feat_map(sam3_predictor.features)  # (C, H', W')

        result = []
        for mask, class_id in zip(gt_masks, gt_class_ids):
            if mask.astype(bool).sum() == 0:
                _record_skip(skip_log, image_id, class_id, "empty GT mask (0 pixels)")
                continue
            emb = _mask_pool_embedding(feat_map, mask)
            if emb is None:
                _record_skip(
                    skip_log, image_id, class_id,
                    "mask collapsed to 0 feature cells (object too small relative to the "
                    "SAM3 feature grid resolution)",
                )
                continue
            emb = F.normalize(emb, p=2, dim=-1).detach().cpu().numpy()
            result.append({"embedding": emb, "class_id": class_id})
        batch_results.append(result)
    return batch_results


def embed_gt_masks(model_name, model, images, batch_gt_masks, batch_gt_class_ids, device,
                    batch_image_ids=None, skip_log=None, **kwargs):
    """
    Embed ground-truth instance mask crops with a visual encoder.

    images: list of np.ndarray (H, W, 3), one per image in the batch.
    batch_gt_masks: list (per image) of GT instance masks (H, W) arrays.
    batch_gt_class_ids: list (per image) of class ids, parallel to batch_gt_masks.
    batch_image_ids: optional list of identifiers (e.g. image path), parallel
        to `images`, used only to attribute skipped instances in `skip_log`.
    skip_log: optional list (mutated in place) that every GT instance which
        could not be embedded gets appended to, as
        {"image_id": ..., "class_id": ..., "reason": ...}. If None, skipped
        instances are dropped silently (matches the old behavior). This is
        the mechanism for surfacing "gsam embedded 19/22, sam3 embedded
        21/22" style discrepancies instead of a bare final count.
    **kwargs: accepted and ignored, so run_experiment2.py can pass
        environment-specific extras (e.g. clipdino's `global_vocab`)
        uniformly across all three *_masks_helper.py modules.

    For "gsam"/"sam3", `model` is expected to be exactly what
    `init_model()` returns for those names ([groundingdino_model,
    sam_predictor] for gsam; the SAM3SemanticPredictor itself for sam3).
    "gsam" is always embedded via GroundingDINO's own Swin backbone
    (text-independent, since it's pulled before the feature-enhancer's
    cross-modal fusion) -- SAM's decoder never sees class identity, so
    it isn't used here.

    Returns a list (per image) of lists of {"embedding": np.ndarray, "class_id": int}.
    """
    if model_name == "clip":
        clip_model = model[1]
        return _embed_crop_based(
            lambda pil_imgs: _clip_embed_crops(clip_model, pil_imgs, device),
            images, batch_gt_masks, batch_gt_class_ids,
            batch_image_ids=batch_image_ids, skip_log=skip_log,
        )
    elif model_name == "siglip":
        siglip_processor, siglip_model = model[1], model[2]
        return _embed_crop_based(
            lambda pil_imgs: _siglip_embed_crops(siglip_processor, siglip_model, pil_imgs, device),
            images, batch_gt_masks, batch_gt_class_ids,
            batch_image_ids=batch_image_ids, skip_log=skip_log,
        )
    elif model_name == "gsam":
        # GroundingDINO's own Swin backbone only -- this is GSAM's detection
        # half, the part that actually recognizes objects (SAM's decoder
        # never sees class identity). Pulled before the feature-enhancer's
        # text cross-attention fusion, so it's independent of whatever
        # class prompt happens to be active.
        groundingdino_model = model[0]
        return _groundingdino_embed_gt_masks(
            groundingdino_model, images, batch_gt_masks, batch_gt_class_ids, device,
            batch_image_ids=batch_image_ids, skip_log=skip_log,
        )
    elif model_name == "sam3":
        return _sam3_embed_gt_masks(
            model, images, batch_gt_masks, batch_gt_class_ids,
            batch_image_ids=batch_image_ids, skip_log=skip_log,
        )
    else:
        raise ValueError(
            f"Model '{model_name}' does not support GT-mask embedding extraction. "
            f"Choose from ['clip', 'siglip', 'gsam', 'sam3'] -- 'yoloe' doesn't "
            f"currently expose a usable embedding API in this codebase."
        )


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