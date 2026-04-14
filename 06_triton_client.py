"""
Step 6: Call Triton Inference Server
=====================================

What this teaches:
    - How to send requests to Triton (HTTP and gRPC)
    - How to preprocess images and send them
    - How to compare embeddings for ReID matching
    - Benchmarking Triton throughput

Triton exposes 3 ports:
    8000 = HTTP  (REST API, easy to use, slightly slower)
    8001 = gRPC  (binary protocol, faster, used in production)
    8002 = Metrics (Prometheus format)

Usage:
    # Make sure Triton is running (step 05)

    # Basic inference
    python 06_triton_client.py --image person.jpg

    # Compare two images (ReID matching)
    python 06_triton_client.py --image1 person_cam1.jpg --image2 person_cam2.jpg

    # Benchmark
    python 06_triton_client.py --benchmark

    # Check server health
    python 06_triton_client.py --health
"""

import numpy as np
import argparse
import time
import os

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
except ImportError:
    print("Install: pip install tritonclient[all]")
    exit(1)

from PIL import Image
from torchvision import transforms


MODEL_NAME = "clip_reid"
TRITON_HTTP_URL = "localhost:8000"
TRITON_GRPC_URL = "localhost:8001"


def get_transform():
    """Same preprocessing as training."""
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def preprocess_image(image_path):
    """Load and preprocess image to numpy array."""
    transform = get_transform()
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).numpy()  # (3, 256, 128)
    return tensor.astype(np.float32)


def check_health(url=TRITON_HTTP_URL):
    """Check if Triton server is ready."""
    try:
        client = httpclient.InferenceServerClient(url=url)
        if client.is_server_ready():
            print("Triton server: READY")
            models = client.get_model_repository_index()
            print(f"Models loaded:")
            for model in models:
                print(f"  {model['name']} (state: {model.get('state', '?')})")
            return True
        else:
            print("Triton server: NOT READY")
            return False
    except Exception as e:
        print(f"Cannot connect to Triton at {url}: {e}")
        return False


def infer_http(input_array, url=TRITON_HTTP_URL):
    """
    Send inference request via HTTP.
    Simple, works with curl too, slightly slower than gRPC.
    """
    client = httpclient.InferenceServerClient(url=url)

    # Prepare input
    inputs = [httpclient.InferInput("input", input_array.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_array)

    # Prepare output
    outputs = [httpclient.InferRequestedOutput("embedding")]

    # Send request
    result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

    # Get output
    embedding = result.as_numpy("embedding")
    return embedding


def infer_grpc(input_array, url=TRITON_GRPC_URL):
    """
    Send inference request via gRPC.
    Faster than HTTP — used in production.
    Binary protocol, less overhead.
    """
    client = grpcclient.InferenceServerClient(url=url)

    # Prepare input
    inputs = [grpcclient.InferInput("input", list(input_array.shape), "FP32")]
    inputs[0].set_data_from_numpy(input_array)

    # Prepare output
    outputs = [grpcclient.InferRequestedOutput("embedding")]

    # Send request
    result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

    embedding = result.as_numpy("embedding")
    return embedding


def compare_images(image1_path, image2_path, protocol='http'):
    """
    ReID matching: extract embeddings from two images
    and compute similarity.
    """
    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    infer_fn = infer_http if protocol == 'http' else infer_grpc

    # Add batch dimension
    emb1 = infer_fn(img1[np.newaxis, ...])
    emb2 = infer_fn(img2[np.newaxis, ...])

    # Cosine similarity
    similarity = np.dot(emb1.flatten(), emb2.flatten()) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    print(f"\nReID Comparison:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    print(f"  Cosine similarity: {similarity:.4f}")
    print(f"  Match: {'LIKELY SAME PERSON' if similarity > 0.5 else 'DIFFERENT PEOPLE'}")

    return similarity


def benchmark(num_runs=200, protocol='http'):
    """Benchmark Triton inference speed."""
    dummy = np.random.randn(1, 3, 256, 128).astype(np.float32)
    infer_fn = infer_http if protocol == 'http' else infer_grpc

    # Warmup
    for _ in range(10):
        infer_fn(dummy)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        infer_fn(dummy)
        times.append((time.time() - start) * 1000)

    times = np.array(times)
    print(f"\nTriton Benchmark ({protocol.upper()}, {num_runs} runs):")
    print(f"  Mean:       {times.mean():.2f} ms")
    print(f"  Median:     {np.median(times):.2f} ms")
    print(f"  P95:        {np.percentile(times, 95):.2f} ms")
    print(f"  P99:        {np.percentile(times, 99):.2f} ms")
    print(f"  Throughput: {1000/times.mean():.1f} images/sec")


def main():
    parser = argparse.ArgumentParser(description='Triton Inference Client')
    parser.add_argument('--image', type=str, help='Single image for inference')
    parser.add_argument('--image1', type=str, help='First image for ReID comparison')
    parser.add_argument('--image2', type=str, help='Second image for ReID comparison')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--health', action='store_true', help='Check server health')
    parser.add_argument('--protocol', type=str, default='http',
                        choices=['http', 'grpc'], help='Protocol to use')
    args = parser.parse_args()

    if args.health:
        check_health()
        return

    if args.image1 and args.image2:
        compare_images(args.image1, args.image2, args.protocol)
        return

    if args.benchmark:
        benchmark(protocol=args.protocol)
        return

    # Single image inference
    infer_fn = infer_http if args.protocol == 'http' else infer_grpc

    if args.image and os.path.exists(args.image):
        input_array = preprocess_image(args.image)[np.newaxis, ...]
    else:
        print("No image — using random input")
        input_array = np.random.randn(1, 3, 256, 128).astype(np.float32)

    embedding = infer_fn(input_array)
    print(f"\nResult:")
    print(f"  Input shape:     {input_array.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  First 10 values: {embedding[0][:10].round(4)}")


if __name__ == '__main__':
    main()
