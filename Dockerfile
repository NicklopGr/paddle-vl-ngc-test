# PaddleOCR-VL-1.5 RunPod Serverless Container - CUDA 11.8 (Ampere sm_86 support)
#
# PURPOSE: Test if CUDA 11.8 base image supports A40/A5000 GPU for cv worker
# CUDA 11.8 includes sm_86 kernels for Ampere, while CUDA 12.x dropped them.
#
# Architecture:
#   paddleocr genai_server (background, port 8080) - vLLM backend with PaddleOCR-VL-1.5-0.9B
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client
#
# KEY DIFFERENCE from production:
#   - Uses CUDA 11.8 base image with sm_86 compiled kernels
#   - device="gpu" for cv worker (PP-DocLayoutV3) to test GPU support

# PaddlePaddle CUDA 11.8 image from Docker Hub (fast download, has sm_86 for Ampere GPUs)
FROM paddlepaddle/paddle:3.0.0rc1-gpu-cuda11.8-cudnn8.6-trt8.5

USER root

WORKDIR /app

# Install system dependencies for OpenCV and PDF processing
# Note: libgl1 works on both Ubuntu 22.04 and 24.04 (libgl1-mesa-glx renamed in 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PaddleOCR with doc-parser support (skip genai-vllm extra - installs wrong CUDA version)
RUN pip install --no-cache-dir "paddleocr[doc-parser]>=3.4.0" "paddlex>=3.4.0"

# Install vLLM CUDA 11.8 wheel (latest vLLM requires CUDA 12.x, we need older version)
# vLLM 0.4.x was the last to support CUDA 11.8
RUN pip install --no-cache-dir "vllm==0.4.3"

# Note: paddleocr install_genai_server_deps cannot run during build (needs libcuda.so)
# It will be run at container startup in start.sh when GPU is available

# Install RunPod SDK
RUN pip install --no-cache-dir runpod requests

# Pre-download models (optional, speeds up cold start)
RUN python -c "from paddleocr import PaddleOCRVL; print('PaddleOCR-VL imports ok')" || true

COPY handler.py /app/
COPY --chmod=755 start.sh /app/

ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV DISABLE_MODEL_SOURCE_CHECK=True
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

CMD ["bash", "/app/start.sh"]
