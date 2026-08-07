# vlm_eval
Workspace for evaluating multiple different VLMs


## Set Up

1. Build Docker Image if not already done: `$ docker compose build`
2. Start up docker container with: `$ docker compose run --rm vision_foundation_models /bin/bash`

## Usage

1. When using CLIP-DINOiser, activate the respective conda environment: `$ activate-clipdino`
2. When using all other models (SAM3, CLIP, Grounding-DINO, Grounded-SAM, and RADIO): `$ activate-main`