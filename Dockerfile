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

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex:paddlex3.3.11-paddlepaddle3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6

USER root

WORKDIR /app

# Install system dependencies for OpenCV and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install FlashAttention first (critical - must be before vLLM)
# vLLM depends on FlashAttention, and installing it after often leads to broken builds
RUN pip install --no-cache-dir flash-attn==2.8.2 --no-build-isolation || \
    echo "FlashAttention wheel install failed, will use vLLM without it"

# Install vLLM genai_server dependencies
# This adds the 'genai_server' subcommand to paddleocr CLI
RUN pip install --no-cache-dir vllm>=0.6.0 && \
    paddleocr install_genai_server_deps vllm || \
    pip install --no-cache-dir "paddleocr[genai-vllm]>=3.4.0"

# Install PaddleOCR with doc-parser support (may already be in base image)
RUN pip install --no-cache-dir "paddleocr[doc-parser]>=3.4.0" "paddlex>=3.4.0" || true

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
