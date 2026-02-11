# PaddleOCR-VL-1.5 RunPod Serverless Container - CUDA 11.8 Test
#
# PURPOSE: Test if we can get cv worker working on GPU with Ampere GPUs
# This version tests device="gpu" for PP-DocLayoutV3 instead of device="cpu"
#
# Architecture:
#   paddleocr genai_server (background, port 8080) - vLLM backend with PaddleOCR-VL-1.5-0.9B
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest

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

# Install PaddlePaddle CPU (layout detection uses CPU, VLM uses vLLM server on GPU)
# CPU version avoids CUDA library conflicts with vLLM
RUN pip install --no-cache-dir paddlepaddle==3.0.0

# Install PaddleOCR with doc-parser support
RUN pip install --no-cache-dir "paddleocr[doc-parser]>=3.4.0" "paddlex>=3.4.0"

# Install RunPod SDK
RUN pip install --no-cache-dir runpod requests

# Pre-download layout model (PP-DocLayoutV3)
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
