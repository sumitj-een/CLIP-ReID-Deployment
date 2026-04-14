"""
Step 2: Export PyTorch model to ONNX
=====================================

What this teaches:
    - Why ONNX exists (universal model format)
    - How torch.onnx.export works (traces the model)
    - What dynamic axes are (variable batch size)
    - How to verify the exported model

The pipeline:
    .pth (PyTorch only) → .onnx (runs anywhere) → TensorRT / Triton / ONNX Runtime

ONNX = Open Neural Network Exchange
Think of it as PDF for models. PyTorch is Word, TensorFlow is Pages.
ONNX is the format everyone can read.

Usage:
    python 02_export_onnx.py --weights path/to/model.pth --output model.onnx

    # Without weights (uses random initialization for demo)
    python 02_export_onnx.py --output model.onnx
"""

import torch
import onnx
import argparse
import os
import sys
import numpy as np

# Import the model from step 1
from load_model_01 import SimpleViTReID, get_transform


def export_to_onnx(model, output_path, device='cpu'):
    """
    Export PyTorch model to ONNX format.

    How it works:
        torch.onnx.export runs a DUMMY input through the model
        and records every operation. The recorded graph becomes
        the ONNX model.

        This is called "tracing" — it traces the execution path.
    """

    model.eval()
    model = model.to(device)

    # Create a dummy input matching the expected shape
    # CLIP-ReID expects: (batch_size, 3, 256, 128)
    dummy_input = torch.randn(1, 3, 256, 128).to(device)

    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input shape: {list(dummy_input.shape)}")

    torch.onnx.export(
        model,                          # the model
        dummy_input,                    # example input (for tracing)
        output_path,                    # output file

        # Name the inputs and outputs (used by Triton later)
        input_names=['input'],
        output_names=['embedding'],

        # Dynamic axes: allow variable batch size
        # Without this, the model only accepts batch_size=1
        dynamic_axes={
            'input': {0: 'batch_size'},       # batch dimension can vary
            'embedding': {0: 'batch_size'},
        },

        opset_version=17,               # ONNX operation set version
        do_constant_folding=True,        # optimize: pre-compute constant expressions
    )

    print(f"  Exported successfully!")

    # ── Verify the ONNX model ──
    print(f"\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)  # raises error if invalid
    print(f"  Model is valid!")

    # Print model info
    print(f"\n{'='*50}")
    print(f"ONNX Model Info:")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Show inputs
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name} → shape {shape}")

    # Show outputs
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} → shape {shape}")

    print(f"{'='*50}")

    return onnx_model


def verify_onnx_matches_pytorch(model, onnx_path, device='cpu'):
    """
    Run the same input through PyTorch and ONNX Runtime.
    Results should be nearly identical (tiny floating point differences are ok).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping verification")
        return

    model.eval()
    model = model.to(device)

    # Same input for both
    test_input = torch.randn(1, 3, 256, 128).to(device)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(test_input).cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path)
    ort_output = session.run(
        ['embedding'],
        {'input': test_input.cpu().numpy()}
    )[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - ort_output))
    print(f"\nVerification: PyTorch vs ONNX Runtime")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Match: {'YES' if max_diff < 1e-4 else 'NO — check export!'}")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to .pth weights')
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Output ONNX file path')
    args = parser.parse_args()

    # Create and load model
    model = SimpleViTReID(num_classes=751, embed_dim=768)

    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights: {args.weights}")
        state_dict = torch.load(args.weights, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No weights — exporting randomly initialized model (for demo)")

    # Export
    export_to_onnx(model, args.output, device='cpu')

    # Verify
    verify_onnx_matches_pytorch(model, args.output, device='cpu')


if __name__ == '__main__':
    main()
