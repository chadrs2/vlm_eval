# vlm_eval
Workspace for evaluating multiple different VLMs


## Set Up

1. Build Docker Image if not already done: `$ docker compose build`
2. Start up docker container with: `$ docker compose run --rm vision_foundation_models bash`

## Usage


### Conda 

1. When using CLIP-DINOiser, activate the respective conda environment: `$ activate-clipdino`
2. When using all other models (SAM3, CLIP, Grounding-DINO, Grounded-SAM, and RADIO): `$ activate-main`

### Hugging Face

1. Log into hugging face within the docker container in the `activate-main` conda environment with: `$ hf auth login` OR `$ huggingface-cli login`
   1. Paste a generated authorization token from your hugging face account
2. To access SAM3 model: Request access for the SAM3 model at: https://huggingface.co/facebook/sam3