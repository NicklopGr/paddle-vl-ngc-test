# PaddleOCR-VL NVIDIA NGC Test

**Purpose:** Test if NVIDIA NGC PaddlePaddle container supports A40 GPU for cv worker (PP-DocLayoutV3).

## Background

The production container uses Baidu's base image which has CUDA kernels that don't support A40 GPU (compute capability 8.6). This causes cv worker crashes:

```
Exception from the 'cv' worker: std::exception
CUDA error 209: no kernel image is available for execution on the device
```

This test container uses NVIDIA's official NGC PaddlePaddle image which should have broader GPU support.

## Key Differences from Production

| Aspect | Production | This Test |
|--------|------------|-----------|
| Base Image | `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest` | `nvcr.io/nvidia/paddlepaddle:24.12-py3` |
| cv worker device | `device="cpu"` (workaround) | `device="gpu"` (testing) |
| Expected Result | Works but uses CPU for layout | May work on A40 GPU |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_DEVICE` | `gpu` | Set to `cpu` if GPU mode crashes |
| `PADDLEX_HOME` | `/runpod-volume/paddlex_models` | Model cache location |

## Testing

### Build Locally

```bash
docker build -t paddle-vl-ngc-test .
```

### Deploy to RunPod

1. Push to Docker Hub or GHCR
2. Create new RunPod Serverless endpoint with this image
3. Check logs for:
   - If GPU mode works: `cv_device=gpu` and no cv worker errors
   - If GPU mode fails: `*** CV WORKER CUDA ERROR DETECTED ***`

### Test Requests

```bash
# Warmup
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"warmup": true}}'

# Process image
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"image_urls": ["https://example.com/page1.png"]}}'
```

## Expected Outcomes

### Success (NGC fixes A40 support)

```
[PaddleOCR-VL] CV_DEVICE=gpu (cv worker will use GPU)
[PaddleOCR-VL] Pipeline loaded in X.XXs (vLLM v1.5 backend, cv_device=gpu)
[PaddleOCR-VL] Warmup inference done in X.XXs
```

### Failure (NGC does NOT fix A40 support)

```
[PaddleOCR-VL] CV_DEVICE=gpu (cv worker will use GPU)
[PaddleOCR-VL] *** CV WORKER CUDA ERROR DETECTED ***
[PaddleOCR-VL] This means NGC container does NOT fix the A40 compatibility issue
[PaddleOCR-VL] Set CV_DEVICE=cpu to use CPU mode instead
```

## Fallback

If GPU mode fails, set `CV_DEVICE=cpu` in RunPod environment variables to fall back to CPU mode (same as production).

# Build trigger 23:41:58
# Build 00:04:45
