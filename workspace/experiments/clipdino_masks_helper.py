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
        
        # Convert logits to probabilities to calculate confidence scores
        probs_np = F.softmax(output[0], dim=0).numpy()
        class_map = probs_np.argmax(axis=0)
 
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
            
            # Calculate instance score as mean probability of the masked pixels
            if binary_mask.any():
                score = float(probs_np[vocab_idx][binary_mask].mean())
            else:
                score = 0.0
 
            if gt_class_id in masks:
                masks[gt_class_id].append((mask_resized, score))
            else:
                masks[gt_class_id] = [(mask_resized, score)]
 
        batch_masks.append(masks)
 
    return batch_masks

# ------------------------------------------------------------------------------------------
# ------------------------------- GT Mask Embeddings (Experiment 2) ------------------------
# ------------------------------------------------------------------------------------------
#
# CLIP-DINOiser doesn't expose a raw "encode this crop" function the way
# CLIP/SigLIP do in gsam_masks_helper.py -- the only public output of this
# model is a per-pixel similarity map against a text vocabulary (see
# `_predict_clipdino` above). So rather than guess at an internal attribute
# name to pull raw CLIP features out of `model.clip_backbone` (which I can't
# verify against your actual installed version), each GT mask is embedded as
# its *mean similarity vector against a large fixed vocabulary*, pooled over
# the mask footprint and L2-normalized. This reuses only the forward-pass
# code paths `_predict_clipdino` already exercises successfully.
#
# This is a genuinely different kind of embedding than CLIP/SigLIP's raw
# visual features in gsam_masks_helper.py -- same/different-class masks
# should still cluster by their similarity *profile* across the vocabulary,
# but don't over-interpret a direct top-k comparison against clip/siglip as
# apples-to-apples; they're different embedding spaces by construction.

def _run_clipdino_forward(model, image, vocab, device, apply_found=True):
    """Single forward pass producing a dense per-pixel probability map
    over `vocab`, same as the core of `_predict_clipdino` above."""
    class_offset = 1 if apply_found else 0
    full_vocab = (["background"] + list(vocab)) if apply_found else list(vocab)

    model.clip_backbone.decode_head.update_vocab(full_vocab)
    model.apply_found = apply_found
    model.to(device)

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

    probs = F.softmax(output[0], dim=0).numpy()  # (C, H, W)
    return probs, class_offset


def embed_gt_masks(model_name, model, images, batch_gt_masks, batch_gt_class_ids, device,
                    global_vocab=None, **kwargs):
    """
    Embed ground-truth instance masks using CLIP-DINOiser's own per-pixel
    class-similarity output, mean-pooled within each mask.

    global_vocab: the fixed vocabulary every mask is scored against.
    REQUIRED -- pass the full set of dataset class names (not just the
    current query classes) so the resulting vectors are as discriminative
    as possible. run_experiment2.py supplies this automatically from the
    dataset's YAML.
    **kwargs: accepted and ignored (e.g. gsam's `model_name` routing isn't
    needed here since this file only ever loads one model).
    """
    if not global_vocab:
        raise ValueError(
            "embed_gt_masks for clipdino requires `global_vocab` (a list of class "
            "names to score every mask against) -- pass the dataset's full class list."
        )

    batch_results = []
    for image, gt_masks, gt_class_ids in zip(images, batch_gt_masks, batch_gt_class_ids):
        if len(gt_masks) == 0:
            batch_results.append([])
            continue

        probs, _ = _run_clipdino_forward(model, image, global_vocab, device, apply_found=True)  # (C, H, W)

        instances = []
        for mask, class_id in zip(gt_masks, gt_class_ids):
            mask_bool = mask.astype(bool)
            if mask_bool.sum() == 0:
                continue

            emb = probs[:, mask_bool].mean(axis=1)  # (C,)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            instances.append({"embedding": emb, "class_id": class_id})

        batch_results.append(instances)

    return batch_results


# ------------------------------------------------------------------------------------------
# ----------------------------- Text-Class Embeddings (Experiment 3) -----------------------
# ------------------------------------------------------------------------------------------
#
# CLIP-DINOiser has no raw text-encoder call exposed the way CLIP/SigLIP do
# (see the note above embed_gt_masks), so there's no way to build a "real"
# text embedding here. But embed_gt_masks above already embeds a GT mask as
# its mean *per-vocab-class probability*, pooled over the mask -- so the
# self-consistent word-bank entry for class c is simply the standard basis
# (one-hot) vector at class c's position in that SAME vocabulary. Cosine
# similarity between a mask's pooled-probability embedding and class c's
# one-hot row then reduces to exactly the mask's mean predicted probability
# of being class c (up to the mask embedding's own L2 norm) -- i.e.
# ranking word-bank classes by similarity is just ranking classes by the
# model's own per-pixel classification, which is the only notion of
# "text embedding" this architecture actually supports.
#
# `_run_clipdino_forward` always builds `full_vocab = ["background"] + vocab`
# when called with apply_found=True (which embed_gt_masks always does), so
# index 0 of every pooled GT-mask embedding is the background channel and
# class i of the vocab lives at index i+1. embed_text_classes reproduces
# that exact offset so its one-hot rows line up with embed_gt_masks's
# pooled probability vectors dimension-for-dimension.
#
# CRITICAL: this only lands in the same space as your embedding pool if
# `class_names` here is IDENTICAL -- same classes, same order -- to the
# `global_vocab` passed into embed_gt_masks/run_experiment when that pool
# was built. If they differ, index i means a different class in the word
# bank than it does in the pool, and every similarity score is silently
# wrong (not an error -- just meaningless numbers that still run). This is
# NOT enforced or checked here since embed_text_classes has no visibility
# into what global_vocab another call used; run_experiment3.py enforces it
# by passing the exact same list as both `global_vocab` and
# `text_class_names` -- don't change one without the other.

def embed_text_classes(model_name, model, class_names, device, apply_found=True):
    """
    Build a one-hot word bank for Experiment 3 in the same
    background-offset vocab space embed_gt_masks pools its GT-mask
    embeddings into. See the module note above for why a one-hot vector is
    the correct (and only self-consistent) "text embedding" for this
    architecture, and the critical ordering requirement vs. `global_vocab`.

    `model`/`device` are accepted for API parity with the other helpers'
    embed_text_classes but unused here -- this is a pure construction, no
    forward pass needed.

    Returns an (len(class_names), len(class_names) + 1) np.ndarray -- one
    row per class, a 1.0 at that class's background-offset vocab index and
    0 elsewhere. Already unit-norm (one-hot), so no separate normalization
    step is needed.
    """
    n = len(class_names)
    offset = 1 if apply_found else 0
    embeddings = np.zeros((n, n + offset), dtype=np.float64)
    for i in range(n):
        embeddings[i, i + offset] = 1.0
    return embeddings


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