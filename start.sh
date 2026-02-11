#!/bin/bash
set -e

echo "[start.sh] NVIDIA NGC PaddlePaddle Test Container"
echo "[start.sh] Testing A40 GPU support for cv worker (PP-DocLayoutV3)"

export DISABLE_MODEL_SOURCE_CHECK=True

# Use network volume for model cache (faster cold starts)
VOLUME_PATH="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
if [ -d "$VOLUME_PATH" ]; then
  # PADDLEX_HOME is the correct env var for PaddleX model caching
  export PADDLEX_HOME="$VOLUME_PATH/paddlex_models"
  export HF_HOME="$VOLUME_PATH/huggingface"
  export HF_HUB_CACHE="$VOLUME_PATH/huggingface/hub"
  mkdir -p "$PADDLEX_HOME" "$HF_HOME" "$HF_HUB_CACHE"
  echo "[start.sh] Using network volume cache: $VOLUME_PATH"
  echo "[start.sh] PADDLEX_HOME=$PADDLEX_HOME"
else
  echo "[start.sh] No network volume found, using container storage"
fi

# Show GPU info for debugging
echo "[start.sh] GPU Information:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || echo "[start.sh] nvidia-smi failed"

# Start PaddleOCR genai server with vLLM backend
# gpu-memory-utilization passed via --backend_config (env var is ignored by vLLM)
echo "[start.sh] Starting PaddleOCR genai_server with vLLM backend (gpu-memory-utilization=0.85)..."
paddleocr genai_server \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --host 0.0.0.0 \
  --port 8080 \
  --backend vllm \
  --backend_config "gpu-memory-utilization=0.85" &
VLM_PID=$!

# Wait for server to be healthy
echo "[start.sh] Waiting for genai_server on port 8080..."
for i in $(seq 1 300); do
  if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "[start.sh] genai_server ready after ${i}s"
    break
  fi
  if [ "$i" -eq 300 ]; then
    echo "[start.sh] ERROR: genai_server failed to start within 300s"
    exit 1
  fi
  sleep 1
done

# CV_DEVICE controls whether cv worker uses GPU or CPU
# Default: "gpu" to TEST if NGC container fixes the CUDA kernel issue
# Set to "cpu" via RunPod env var to fall back to CPU mode
echo "[start.sh] CV_DEVICE=${CV_DEVICE:-gpu}"

# Start RunPod handler
python -u /app/handler.py
