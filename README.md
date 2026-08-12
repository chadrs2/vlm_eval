# vlm_eval
Workspace for evaluating multiple different VLMs


## Set Up

1. Build Docker Image if not already done: `$ docker compose build`
2. Allow visual display from the docker container: `$ xhost +local:docker`
3. Start up docker container with: `$ docker compose run --rm vision_foundation_models bash`

## Usage


### Conda 

1. When using CLIP-DINOiser, activate the respective conda environment: `$ conda activate clipdino`
2. When using RADIO, activate the respective conda environment: `$ conda activate radio`
3. When using all other models (SAM3, CLIP, Grounding-DINO, and Grounded-SAM): `$ conda activate gsam`

### Hugging Face

1. Log into hugging face within the docker container in the `conda activate gsam` conda environment with: `$ hf auth login` OR `$ huggingface-cli login`
   1. Paste a generated authorization token from your hugging face account
2. To access SAM3 model: Request access for the SAM3 model at: https://huggingface.co/facebook/sam3

### Individual Model Examples

Simple example implementations of each VLM and masking code is contained in the `/workspace/notebooks` folder. 

Grounded-SAM and Grounding-DINO example code is contained in `/workspace/projects/Grounded-Segment-Anything/grounded_sam.ipynb`

CLIP-DINOiser example code is contained in `/workspace/projects/clip_dinoiser/demo.ipynb`

## Experiments

### Experiment 1: Text Query, Mask Predictions

1. Prompts for every class with "<class>", and VLM outputs segmentation mask
2. Computes per class mIoU, TP, FP, FN, TN, 
3. Compute across dataset (all classes): F-mIoU, AP, Runtime?