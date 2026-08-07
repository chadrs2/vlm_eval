#!/bin/bash
set -e

mkdir -p /workspace/projects
cd /workspace/projects

# Grounding DINO
# if [ ! -d "GroundingDINO" ]; then
#     git clone https://github.com/IDEA-Research/GroundingDINO.git
# fi
cd GroundingDINO && pip install --no-build-isolation -e . && cd ..

# Grounded SAM
# if [ ! -d "Grounded-Segment-Anything" ]; then
#     git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
# fi
cd Grounded-Segment-Anything && pip install --no-build-isolation -e segment_anything && cd ..

# SAM3
# if [ ! -d "sam3" ]; then
#     git clone https://github.com/facebookresearch/sam3.git
# fi
cd sam3 && pip install --no-build-isolation -e . && cd ..

echo "Main environment models installed successfully."