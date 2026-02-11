# PaddleOCR-VL-1.5 RunPod Serverless Container - NVIDIA NGC Test
#
# PURPOSE: Test if NVIDIA NGC PaddlePaddle container supports A40 GPU for cv worker
# This uses NVIDIA's official container which should have broader GPU support.
#
# Architecture:
#   paddleocr genai_server (background, port 8080) - vLLM backend with PaddleOCR-VL-1.5-0.9B
#   handler.py (RunPod serverless) - uses PaddleOCRVL pipeline client
#
# KEY DIFFERENCE from production:
#   - Uses nvcr.io/nvidia/paddlepaddle:24.12-py3 instead of Baidu's image
#   - Tests if device="gpu" works without cv worker crashes on A40

FROM nvcr.io/nvidia/paddlepaddle:24.12-py3

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

# Install vLLM (required for genai_server backend)
# NGC container has PaddlePaddle but not vLLM
RUN pip install --no-cache-dir vllm>=0.6.0

# Install PaddleOCR with doc-parser support
RUN pip install --no-cache-dir "paddleocr[doc-parser]>=3.4.0" "paddlex>=3.4.0"

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
