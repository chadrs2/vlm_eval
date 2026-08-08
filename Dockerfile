FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

############################################################
# System dependencies
############################################################

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    unzip \
    ffmpeg \
    tmux \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

############################################################
# Install Miniconda
############################################################

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda config --remove channels defaults || true && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# CUDA environment variables for package compilation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

############################################################
# Create Conda Environments
############################################################

RUN conda create -y -n gdino python=3.10 pip && \
    conda create -y -n gsam python=3.10 pip && \
    conda create -y -n s3 python=3.10 pip && \
    conda create -y -n clipdino python=3.9 pip

############################################################
# Common Base Dependencies Helper Variables
############################################################

ENV BASE_PYTORCH_DEPS="torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118"
ENV BASE_PYTHON_DEPS="setuptools<81 numpy>=1.26,<2 scipy matplotlib opencv-python pillow tqdm iopath accelerate huggingface_hub safetensors timm einops pycocotools supervision jupyterlab"
COPY requirements_main.txt /tmp/

############################################################
# ENVIRONMENT: Grounding DINO (`gdino`)
############################################################

RUN conda run -n gdino pip install ${BASE_PYTORCH_DEPS} && \
    conda run -n gdino pip install ${BASE_PYTHON_DEPS} addict yapf "transformers==4.29.2" "ftfy==6.1.1" && \
    conda run -n gdino pip install -r /tmp/requirements_main.txt && \
    mkdir -p /workspace/projects && \
    cd /workspace/projects && \
    if [ -d "GroundingDINO" ]; then conda run -n gdino pip install --no-build-isolation -e GroundingDINO; fi

############################################################
# ENVIRONMENT: Grounded SAM (`gsam`)
############################################################

RUN conda run -n gsam pip install ${BASE_PYTORCH_DEPS} && \
    conda run -n gsam pip install ${BASE_PYTHON_DEPS} "transformers==4.36.0" "xformers==0.0.28.post3" && \
    conda run -n gsam pip install -r /tmp/requirements_main.txt && \
    cd /workspace/projects && \
    if [ -d "GroundingDINO" ]; then conda run -n gsam pip install --no-build-isolation -e GroundingDINO; fi && \
    if [ -d "Grounded-Segment-Anything" ]; then conda run -n gsam pip install --no-build-isolation -e Grounded-Segment-Anything/segment_anything; fi

############################################################
# ENVIRONMENT: SAM 3 (`s3`)
############################################################

RUN conda run -n s3 pip install ${BASE_PYTORCH_DEPS} && \
    conda run -n s3 pip install ${BASE_PYTHON_DEPS} transformers ultralytics omegaconf pyyaml "ftfy==6.1.1" && \
    conda run -n s3 pip install -r /tmp/requirements_main.txt && \
    cd /workspace/projects && \
    if [ -d "sam3" ]; then conda run -n s3 pip install --no-build-isolation -e sam3; fi

############################################################
# ENVIRONMENT: CLIP-DINOISER (`clipdino`)
############################################################

RUN conda run -n clipdino pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    conda run -n clipdino pip install openmim && \
    conda run -n clipdino mim install mmengine && \
    conda run -n clipdino mim install "mmcv-full==1.6.0" && \
    conda run -n clipdino mim install "mmsegmentation==0.27.0" && \
    conda run -n clipdino pip install open_clip_torch --no-deps && \
    conda run -n clipdino pip install numpy scipy opencv-python pillow tqdm "requests>=2.28.2,<2.29.0" matplotlib pyyaml jupyter && \
    conda run -n clipdino pip install ftfy regex omegaconf && \
    cd /workspace/projects && \
    if [ -d "clip_dinoiser" ]; then conda run -n clipdino pip install --no-build-isolation -e clip_dinoiser; fi

############################################################
# Workspace Setup & Convenience Aliases
############################################################

RUN mkdir -p /workspace/models /workspace/datasets /workspace/notebooks /workspace/projects

ENV MODEL_DIR=/workspace/models
ENV DATASET_DIR=/workspace/datasets

# Convenience aliases to switch environments in bash
RUN echo 'alias activate-gdino="source /opt/conda/etc/profile.d/conda.sh && conda activate gdino"' >> /root/.bashrc && \
    echo 'alias activate-gsam="source /opt/conda/etc/profile.d/conda.sh && conda activate gsam"' >> /root/.bashrc && \
    echo 'alias activate-s3="source /opt/conda/etc/profile.d/conda.sh && conda activate s3"' >> /root/.bashrc && \
    echo 'alias activate-clipdino="source /opt/conda/etc/profile.d/conda.sh && conda activate clipdino"' >> /root/.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]