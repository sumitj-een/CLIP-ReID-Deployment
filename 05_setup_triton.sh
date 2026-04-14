#!/bin/bash
# ============================================================
# Step 5: Set up and run Triton Inference Server
# ============================================================
#
# What this teaches:
#   - Triton is a MODEL SERVER — it serves models via HTTP/gRPC
#   - You don't write a FastAPI app. Triton IS the API.
#   - You just put the model in the right folder structure
#
# Think of it like:
#   FastAPI approach: you write the server + model loading + preprocessing
#   Triton approach:  you give Triton the model, it does everything else
#
# Triton handles:
#   - Model loading/unloading
#   - HTTP and gRPC endpoints
#   - Dynamic batching
#   - Multi-model serving
#   - GPU memory management
#   - Health checks
#   - Metrics (Prometheus)
#
# Prerequisites:
#   1. Export your model: python 02_export_onnx.py --output 05_triton_repo/clip_reid/1/model.onnx
#   2. Docker installed with NVIDIA Container Toolkit (for GPU)
#
# ============================================================

# Step 1: Copy your ONNX model to the Triton repo
echo "Step 1: Ensure model is in the right place"
echo "  Expected: 05_triton_repo/clip_reid/1/model.onnx"

if [ ! -f "05_triton_repo/clip_reid/1/model.onnx" ]; then
    echo "  Model not found! Run first:"
    echo "    python 02_export_onnx.py --output 05_triton_repo/clip_reid/1/model.onnx"
    exit 1
fi

echo "  Model found!"
echo ""

# Step 2: Start Triton server with Docker
echo "Step 2: Starting Triton Inference Server"
echo ""

# GPU version (production)
echo "GPU command:"
echo "  docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\"
echo "    -v $(pwd)/05_triton_repo:/models \\"
echo "    nvcr.io/nvidia/tritonserver:24.01-py3 \\"
echo "    tritonserver --model-repository=/models"
echo ""

# CPU version (testing without GPU)
echo "CPU command:"
echo "  docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\"
echo "    -v $(pwd)/05_triton_repo:/models \\"
echo "    nvcr.io/nvidia/tritonserver:24.01-py3 \\"
echo "    tritonserver --model-repository=/models"
echo ""

# Ports:
#   8000 = HTTP  (REST API)
#   8001 = gRPC  (faster, for production)
#   8002 = Metrics (Prometheus)

echo "Triton endpoints:"
echo "  Health:  curl localhost:8000/v2/health/ready"
echo "  Models:  curl localhost:8000/v2/models"
echo "  Infer:   python 06_triton_client.py"
echo "  Metrics: curl localhost:8002/metrics"
echo ""

# Step 3: Run Triton (uncomment the version you want)
echo "Starting Triton (GPU mode)..."
docker run --gpus all --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$(pwd)/05_triton_repo:/models" \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
