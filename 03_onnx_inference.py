"""
Step 3: Run inference with ONNX Runtime
========================================

What this teaches:
    - ONNX Runtime is FASTER than raw PyTorch for inference
    - It works on CPU and GPU without changing code
    - This is what many production systems use (before Triton)

Speed comparison:
    PyTorch:      ~50ms per image (with overhead)
    ONNX Runtime: ~20ms per image (optimized graph)
    TensorRT:     ~5ms per image (GPU-optimized, step 04)
###
    "The speedup comes mainly from graph-level optimizations and removing training overhead.
For example, operations like Conv → BatchNorm → ReLU can be fused into a single operation, reducing memory access and computation steps.
ONNX Runtime also executes a static graph without Python or autograd, so inference becomes more efficient."
Usage:
    # First export the model (step 02)
    python 02_export_onnx.py --output model.onnx

    # Then run inference
    python 03_onnx_inference.py --model model.onnx --image person.jpg

    # Benchmark speed
    python 03_onnx_inference.py --model model.onnx --benchmark
"""

import numpy as np
import argparse
import os
import time

try:
    import onnxruntime as ort
except ImportError:
    print("Install: pip install onnxruntime-gpu")
    exit(1)

from PIL import Image
from torchvision import transforms


def get_transform():
    """Same preprocessing as PyTorch model."""
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def create_session(model_path):
    """
    Create an ONNX Runtime inference session.
    Automatically uses GPU if available, falls back to CPU.
    """
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
        print("Using: GPU (CUDA)")
    else:
        print("Using: CPU")
    providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(model_path, providers=providers)
    return session


def run_inference(session, input_array):
    """
    Run inference on a numpy array.
    No PyTorch needed — just numpy in, numpy out.
    """
    input_name = session.get_inputs()[0].name    # 'input'
    output_name = session.get_outputs()[0].name  # 'embedding'

    result = session.run(
        [output_name],
        {input_name: input_array}
    )
    return result[0]


def benchmark(session, num_runs=100):
    """
    Measure inference speed.
    First run is always slower (warmup), so we exclude it.
    """
    dummy = np.random.randn(1, 3, 256, 128).astype(np.float32)

    # Warmup
    for _ in range(5):
        run_inference(session, dummy)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        run_inference(session, dummy)
        times.append((time.time() - start) * 1000)  # ms

    times = np.array(times)
    print(f"\nBenchmark ({num_runs} runs):")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Throughput: {1000/times.mean():.1f} images/sec")


def main():
    parser = argparse.ArgumentParser(description='ONNX Runtime Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to .onnx model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to person image')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    args = parser.parse_args()

    # Create session
    session = create_session(args.model)

    # Show model info
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"Model: {args.model}")
    print(f"  Input:  {inp.name} → {inp.shape}")
    print(f"  Output: {out.name} → {out.shape}")

    # Prepare input
    if args.image and os.path.exists(args.image):
        transform = get_transform()
        img = Image.open(args.image).convert('RGB')
        input_array = transform(img).unsqueeze(0).numpy()
    else:
        print("No image — using random input")
        input_array = np.random.randn(1, 3, 256, 128).astype(np.float32)

    # Run inference
    embedding = run_inference(session, input_array)

    print(f"\nResult:")
    print(f"  Input shape:     {input_array.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  First 10 values: {embedding[0][:10].round(4)}")

    # Batch inference demo
    print(f"\nBatch inference (4 images at once):")
    batch = np.random.randn(4, 3, 256, 128).astype(np.float32)
    embeddings = run_inference(session, batch)
    print(f"  Input:  {batch.shape}")
    print(f"  Output: {embeddings.shape}")

    # Benchmark
    if args.benchmark:
        benchmark(session)


if __name__ == '__main__':
    main()
