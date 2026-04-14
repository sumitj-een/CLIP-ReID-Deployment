"""
Step 4: Convert ONNX to TensorRT
==================================

What this teaches:
    - TensorRT is NVIDIA's optimizer for GPU inference
    - Takes an ONNX model and creates a GPU-optimized "engine"
    - 3-10x faster than PyTorch on the same GPU
    - REQUIRES an NVIDIA GPU (run this on your office server)

How TensorRT optimizes:
    1. Fuses layers (conv + batchnorm + relu → one operation)
    2. Selects best GPU kernels for your specific GPU
    3. Reduces precision (FP32 → FP16 → INT8) with minimal accuracy loss
    4. Optimizes memory layout for GPU cache

Speed progression:
    PyTorch:      ~50ms per image
    ONNX Runtime: ~20ms per image
    TensorRT FP32: ~10ms per image
    TensorRT FP16:  ~5ms per image
    TensorRT INT8:  ~2ms per image

Usage (on GPU server):
    # Convert ONNX → TensorRT engine
    python 04_tensorrt_convert.py --onnx model.onnx --output model.engine

    # With FP16 precision (recommended — 2x faster, minimal accuracy loss)
    python 04_tensorrt_convert.py --onnx model.onnx --output model.engine --fp16

    # Alternative: use trtexec CLI tool (comes with TensorRT)
    trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
"""

import argparse
import os
import sys
import time

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT not installed.")
    print("This script requires an NVIDIA GPU and TensorRT.")
    print("")
    print("Install options:")
    print("  1. Use NVIDIA Docker: nvcr.io/nvidia/tensorrt:24.01-py3")
    print("  2. pip install tensorrt (if CUDA is set up)")
    print("  3. Use trtexec CLI: trtexec --onnx=model.onnx --saveEngine=model.engine --fp16")
    sys.exit(1)


# TensorRT logger (shows warnings and errors during conversion)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, output_path, fp16=False, max_batch_size=8):
    """
    Convert ONNX model to TensorRT engine.

    This is a one-time operation. The resulting .engine file is
    specific to YOUR GPU. An engine built on an A100 won't work on a T4.
    """
    print(f"Building TensorRT engine from: {onnx_path}")
    print(f"  FP16 precision: {fp16}")
    print(f"  Max batch size: {max_batch_size}")

    # ── Step 1: Create builder ──
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ── Step 2: Parse ONNX model ──
    print("  Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")

    # ── Step 3: Configure optimization ──
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  Enabled FP16 precision")

    # Set dynamic batch size range
    profile = builder.create_optimization_profile()
    profile.set_shape(
        'input',
        min=(1, 3, 256, 128),             # minimum batch size
        opt=(4, 3, 256, 128),             # optimal batch size (most common)
        max=(max_batch_size, 3, 256, 128)  # maximum batch size
    )
    config.add_optimization_profile(profile)

    # ── Step 4: Build engine (this takes a few minutes) ──
    print("  Building engine (this may take 2-5 minutes)...")
    start = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.time() - start

    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # ── Step 5: Save engine ──
    with open(output_path, 'wb') as f:
        f.write(engine_bytes)

    print(f"\n{'='*50}")
    print(f"TensorRT Engine Built!")
    print(f"  Output: {output_path}")
    print(f"  Size:   {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"  Build time: {build_time:.1f}s")
    print(f"  Precision: {'FP16' if fp16 else 'FP32'}")
    print(f"{'='*50}")


def benchmark_engine(engine_path, num_runs=100):
    """Benchmark TensorRT engine speed."""
    import numpy as np

    print(f"\nBenchmarking: {engine_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape('input', (1, 3, 256, 128))

    # Allocate GPU memory
    import cuda  # pycuda or cuda-python
    input_data = np.random.randn(1, 3, 256, 128).astype(np.float32)

    # Warmup
    for _ in range(10):
        context.execute_v2(bindings=[input_data.ctypes.data])

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        context.execute_v2(bindings=[input_data.ctypes.data])
        times.append((time.time() - start) * 1000)

    times = np.array(times)
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Throughput: {1000/times.mean():.1f} images/sec")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--output', type=str, default='model.engine',
                        help='Output TensorRT engine path')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision (recommended)')
    parser.add_argument('--max-batch', type=int, default=8,
                        help='Maximum batch size')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark after building')
    args = parser.parse_args()

    build_engine(args.onnx, args.output, fp16=args.fp16,
                 max_batch_size=args.max_batch)

    if args.benchmark:
        benchmark_engine(args.output)


if __name__ == '__main__':
    main()
