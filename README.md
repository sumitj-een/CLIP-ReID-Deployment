# Model Serving — From .pth to Production

## The Pipeline

```
.pth (PyTorch) → .onnx (universal) → TensorRT (GPU optimized) → Triton (serving) → K8s (deployed)
     01              02                    04                       05-06              07
```

## Learning Path

| # | File | What You Learn |
|---|------|----------------|
| 1 | `01_load_model.py` | Load .pth weights, run PyTorch inference, extract embeddings |
| 2 | `02_export_onnx.py` | Convert PyTorch → ONNX (universal format) |
| 3 | `03_onnx_inference.py` | Run inference with ONNX Runtime, benchmark speed |
| 4 | `04_tensorrt_convert.py` | Convert ONNX → TensorRT engine (GPU optimization) |
| 5 | `05_triton_repo/` | Triton model repository structure + config |
| 6 | `06_triton_client.py` | Call Triton server (HTTP + gRPC), ReID matching |
| 7 | `07_deploy_k8s/` | Deploy Triton on Kubernetes with GPU |

## Quick Start (on GPU server)

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Export model to ONNX
python 02_export_onnx.py --output 05_triton_repo/clip_reid/1/model.onnx

# Step 3: Start Triton
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/05_triton_repo:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models

# Step 4: Run inference
python 06_triton_client.py --health
python 06_triton_client.py --benchmark
```

## Speed Progression

```
PyTorch:       ~50ms per image
ONNX Runtime:  ~20ms per image
TensorRT FP16:  ~5ms per image
Triton + TRT:   ~5ms + batching = thousands of images/sec
```
