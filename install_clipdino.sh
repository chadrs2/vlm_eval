#!/bin/bash
set -e

mkdir -p /workspace/projects
cd /workspace/projects

# CLIP-DINOiser
# if [ ! -d "clip_dinoiser" ]; then
#     git clone https://github.com/wysoczanska/clip_dinoiser.git
# fi
cd clip_dinoiser && pip install --no-build-isolation -e . && cd ..

echo "CLIP-DINOiser installed successfully."