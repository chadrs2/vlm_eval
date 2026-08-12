import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import cv2
import matplotlib.gridspec as gridspec

PROJECTS_ROOT = Path("/workspace/projects")
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

try:
    from nvidia_radio.hubconf import radio_model
    from nvidia_radio.radio.pamr import PAMR
except ImportError:
    from projects.nvidia_radio.hubconf import radio_model
    from projects.nvidia_radio.radio.pamr import PAMR



#model_version="radio_v2.5-g" # for RADIOv2.5-g model (ViT-H/14)
# model_version="radio_v2.5-h" # for RADIOv2.5-H model (ViT-H/16)
# model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
# model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
model_version="c-radio_v3-b" # for C_RADIOv3-B model (ViT-B/16)
# model_version="e-radio_v2" # for E-RADIO
adaptor_version="siglip2" # ["clip", siglip", "siglip2"] for v2 and v3 models

#### v4 Models- Currently not working, need to edit radio_model loading code ####
# model_version = "c-radio_v4-so400m" # for C-RADIOv4-SO400M model (ViT-B/16)
# model_version = "c-radio_v4-h" # for C-RADIOv4-H model (ViT-H/16)
# adaptor_version = "siglip2-g" # ['siglip2-g', 'dino_v3_7b', 'sam3'] for v4 models


print(f"Loading {model_version} with {adaptor_version} adapter...")
model, chk = radio_model(
    version=model_version,
    progress=True,
    skip_validation=True,
    adaptor_names=adaptor_version,
    return_checkpoint=True, 
    use_naclip=True, 
    naclip_strategy="kkonly", #"kkonly",
    naclip_gaussian_std=5.0,
    fixed_patch_dim=(40,40), #(45,80),
    gaussian_device='cuda',
    use_summary_for_spatial=True
)


size_model = 0
for param in model.parameters():
    if param.data.is_floating_point():
        size_model += param.numel() * torch.finfo(param.data.dtype).bits
    else:
        size_model += param.numel() * torch.iinfo(param.data.dtype).bits
print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")


model.cuda().eval() 

def resize_long_side(img, max_side=1024):
    w, h = img.size

    if max(w, h) <= max_side:
        return img

    scale = max_side / max(w, h)

    return img.resize(
        (int(w * scale), int(h * scale)),
        Image.LANCZOS
    )

# img_path = '/Data/cradio_v4.png'
# img_path = '/Data/coastal_test_img.jpg'
# img_path = '/Data/lab_picture.jpg'
# img_path = '/Data/lab_picture_resized.jpg'
img_path = '/workspace/projects/Grounded-Segment-Anything/assets/inpaint_demo.jpg'
x = Image.open(img_path).convert('RGB')
# x = resize_long_side(x, max_side=1024)
img = x.copy()

x = pil_to_tensor(x).to(dtype=torch.float32, device='cuda')
x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
x = x.unsqueeze(0) # Add a batch dimension


nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

print(f"Original input shape: {x.shape}")
print(f"Nearest supported resolution: {nearest_res}")
print(f"Model Input shape: {x.shape}")

if "e-radio" in model_version:
    model.model.set_optimal_window_size(x.shape[2:]) #where it expects a tuple of (height, width) of the input image.

# text_queries = [
#     "water", "bridge", "sky", "vegetation", "boat"
# ] 
# text_queries = ["engineer", "student", "professor", "happiness", "sadness"]
# text_queries = ["old", "young", "graduated"]
# text_queries = ["happy", "sad", "engineer"]
text_queries = ["a fluffy dog", "a bench"]

with torch.no_grad():
    adaptor = model.adaptors[adaptor_version]
    tokens = adaptor.tokenizer(text_queries).to('cuda')
    text_feats = adaptor.encode_text(tokens, normalize=True) # Shape: [K, Text_Dim]

    # summary, spatial_features = model(x, feature_fmt='NCHW')[adaptor_version]
    # assert spatial_features.ndim == 4
    
    # 1. Run model without feature_fmt to bypass the CPE/attn_mask bug
    summary, spatial_features = model(x)[adaptor_version]
    
    # 2. Extract grid dimensions from the RADIO wrapper itself.
    # The wrapper exposes `patch_size` as a property, not a `.config` object.
    patch_size = model.patch_size
    grid_h = x.shape[2] // patch_size
    grid_w = x.shape[3] // patch_size
    
    # 3. Manually reshape spatial_features from (B, T, C) to (B, C, H, W)
    # If spatial_features has shape (B, T, C):
    B, T, C = spatial_features.shape
    spatial_features = spatial_features.permute(0, 2, 1).reshape(B, C, grid_h, grid_w)

    assert spatial_features.ndim == 4
    

print(f"Spatial features shape: {spatial_features.shape}")
print(f"Summary shape: {summary.shape}")


# Visualize cosine similarity between text embeddings and spatial features
spatial_feats = spatial_features[0] # Remove batch dimension, shape: [Channels, Height, Width]
spatial_feats = spatial_feats / spatial_feats.norm(dim=0, keepdim=True) # Normalize spatial features
c, h, w = spatial_feats.shape
spatial_feats = spatial_feats.view(c, h*w) # Reshape to [Channels, Num_Patches]
test_sim_spatial = text_feats @ spatial_feats # Get cosine similarity, shape: [K, Num_Patches]
text_sim_spatial = test_sim_spatial.view(len(text_queries), h, w) # Reshape back to [K, Height_in_Patches, Width_in_Patches]

similarity_grid = text_sim_spatial
num_queries = len(text_queries)
orig_h, orig_w = img.size[1], img.size[0]


pixel_level_seg = True
if pixel_level_seg:
    text_sim_spatial = F.interpolate(
        text_sim_spatial.unsqueeze(0), size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False
    ).squeeze(0)  # (num_texts, H, W)

    # Apply PAMR to the text similarity map (mask refinement: Patch to Pixels)
    # if GPU size allows, use: 
    # num_iter = 50
    # dilations = [1, 2, 4, 8, 12, 24]
    pamr = PAMR(10, dilations=[1, 2]).to('cuda')
    text_sim_spatial = pamr(x*255, text_sim_spatial.unsqueeze(0)).squeeze(0)  # (num_texts, H, W)

pred_labels = text_sim_spatial.argmax(dim=0).cpu().numpy()  # (H, W)
pred_labels_resized = pred_labels

# To get patch level categories, resize the labels
if not pixel_level_seg:
    pred_labels_resized = cv2.resize(
        pred_labels, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_NEAREST
    )

# visualize the predictions
image_res = (x.shape[-2], x.shape[-1])  # (H, W)
num_queries = len(text_queries)
cmap = plt.get_cmap('tab10', num_queries)
color_map = np.array([cmap(i)[:3] for i in range(num_queries)])  # [Q, 3], float [0, 1]

# Create segmap
segmap = np.zeros((image_res[0], image_res[1], 3), dtype=np.float32)
for q in range(num_queries):
    for c in range(3):
        segmap[..., c] += (pred_labels_resized == q) * color_map[q, c]

img_resized = img.resize(
    (x.shape[-1], x.shape[-2]),
    Image.Resampling.LANCZOS
)

img_rgb = np.array(img_resized.convert('RGB')) / 255.0  # Convert to float [0, 1]
segmap = np.clip(segmap, 0, 1)

viz_img = np.concatenate([
    img_rgb, segmap
], axis=1)  # Concatenate original image and segmap side by side
viz_img = (viz_img * 255).astype(np.uint8)


fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.1, wspace=0.9)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Patch-Level Text Alignment")
ax1.imshow(viz_img)
ax1.axis('off')

# Legend below 2nd plot
ax_legend = fig.add_subplot(gs[1, 0])
ax_legend.axis('off')
handles = [plt.Line2D([0], [0], color=color_map[i], lw=6) for i in range(len(text_queries))]
labels = [text_queries[i] for i in range(len(text_queries))]
ax_legend.legend(handles, labels, loc='center', ncol=num_queries, fontsize='small', frameon=False)

plt.show()


fig, axes = plt.subplots(1, num_queries, figsize=(5 * num_queries, 5), squeeze=False)
orig_img = Image.open(img_path)

for i, query in enumerate(text_queries):
    query_heatmap = similarity_grid[i]

    # Smoothly upscale the 2D patch map back to the high-res native pixels.
    # Use a detached clone to avoid the warning from copying a tensor into a new tensor.
    heatmap_tensor = torch.as_tensor(query_heatmap, dtype=torch.float32).detach().unsqueeze(0).unsqueeze(0)
    heatmap_upsampled = F.interpolate(
        heatmap_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()

    ax = axes[0, i]
    ax.imshow(orig_img)
    # High similarity will now pop out as burning red, low as blue
    im = ax.imshow(heatmap_upsampled, cmap='jet', alpha=0.45) 
    ax.set_title(f"Query: '{query}'", fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, fraction=0.046)

plt.tight_layout()
plt.show()