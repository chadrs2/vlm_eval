FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm /tmp/miniconda.sh && \
    conda clean -a -y

# Accept Anaconda ToS to prevent channel errors
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Force pip to use the legacy dependency resolver inside conda
RUN /root/miniconda3/bin/pip config set global.use-deprecated legacy-resolver

WORKDIR /workspace

##################################
## Grounded-SAM + GroundingDINO ##
##################################
# Base environment creation
COPY gsam_environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -a -y

ENV PATH=/root/miniconda3/envs/gsam/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda

# Build submodules from repository
COPY workspace/projects/Grounded-Segment-Anything /tmp/Grounded-Segment-Anything

RUN cd /tmp/Grounded-Segment-Anything/GroundingDINO && \
    pip install --no-build-isolation -e . && \
    cd /tmp/Grounded-Segment-Anything/segment_anything && \
    pip install -e . && \
    rm -rf /tmp/Grounded-Segment-Anything

##################################
## SAM 3, YOLO-E, CLIP, FastSAM ##
##################################
RUN pip install ultralytics

############
## SigLIP ##
############
RUN pip install sentencepiece protobuf

############################################
## Fix PyTorch installation for CUDA 12.1 ##
############################################
# Activate the gsam environment and reinstall PyTorch compiled for CUDA 12.1
RUN pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121


###########
## RADIO ##
###########
# Create the RADIO conda environment
COPY radio_env.yml /tmp/radio_env.yml

RUN conda env create -f /tmp/radio_env.yml && \
    conda clean -a -y

# Copy the RADIO repository
COPY workspace/projects/nvidia_radio /tmp/nvidia_radio

# Install RADIO into the radio environment
RUN rm -rf /tmp/nvidia_radio /tmp/radio_env.yml


###################
## CLIP-DINOiser ##
###################

RUN echo "source activate gsam" >> ~/.bashrc


CMD ["bash"]