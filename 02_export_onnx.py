"""
Step 2: Export CLIP-ReID model to ONNX
=======================================

What this teaches:
    - Why ONNX exists (universal model format)
    - How torch.onnx.export works (traces the model)
    - How to handle CLIP-ReID's specific forward signature
    - How to verify the exported model

The pipeline:
    .pth (PyTorch only) → .onnx (runs anywhere) → TensorRT / Triton

ONNX = Open Neural Network Exchange
Think of it as PDF for models. PyTorch is Word, TensorFlow is Pages.
ONNX is the format everyone can read.

Usage:
    python 02_export_onnx.py --weights path/to/model.pth --output model.onnx
    python 02_export_onnx.py --output model.onnx   # no weights, random init
"""

import torch
import torch.nn as nn
import onnx
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'clip_reid_repo'))

from config import cfg as default_cfg
from model.make_model_clipreid import make_model


class CLIPReIDWrapper(nn.Module):
    """
    Wrapper that simplifies the forward signature for ONNX export.

    CLIP-ReID's forward() has optional kwargs (cam_label, view_label, etc.)
    that ONNX tracing can't handle. This wrapper takes only the image
    tensor and returns the normalized embedding.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        embedding = self.model(x)
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


def get_cfg(img_size=(256, 128), model_name='ViT-B-16', num_classes=751, dataset='market1501'):
    cfg = default_cfg.clone()
    cfg.MODEL.NAME = model_name
    cfg.MODEL.NECK = 'bnneck'
    cfg.MODEL.COS_LAYER = False
    cfg.MODEL.STRIDE_SIZE = [16, 16]
    cfg.MODEL.SIE_COE = 3.0
    cfg.MODEL.SIE_CAMERA = False
    cfg.MODEL.SIE_VIEW = False
    cfg.INPUT.SIZE_TRAIN = list(img_size)
    cfg.INPUT.SIZE_TEST = list(img_size)
    cfg.TEST.NECK_FEAT = 'after'
    cfg.TEST.FEAT_NORM = 'yes'
    cfg.DATASETS.NAMES = dataset
    cfg.freeze()
    return cfg


def export_to_onnx(model, output_path, img_size=(256, 128), device='cpu'):
    """Export CLIP-ReID model to ONNX format."""

    # Wrap the model for clean ONNX export
    wrapper = CLIPReIDWrapper(model)
    wrapper.eval()
    wrapper = wrapper.to(device)

    dummy_input = torch.randn(1, 3, *img_size).to(device)

    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input shape: {list(dummy_input.shape)}")

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"  Exported successfully!")

    # Verify
    print(f"\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Model is valid!")

    # Print info
    print(f"\n{'='*50}")
    print(f"ONNX Model Info:")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name} → shape {shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} → shape {shape}")
    print(f"{'='*50}")


def verify_onnx_matches_pytorch(model, onnx_path, img_size=(256, 128), device='cpu'):
    """Verify ONNX output matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping verification")
        return

    wrapper = CLIPReIDWrapper(model)
    wrapper.eval()
    wrapper = wrapper.to(device)

    test_input = torch.randn(1, 3, *img_size).to(device)

    # PyTorch
    with torch.no_grad():
        pytorch_output = wrapper(test_input).cpu().numpy()

    # ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    ort_output = session.run(['embedding'], {'input': test_input.cpu().numpy()})[0]

    max_diff = np.max(np.abs(pytorch_output - ort_output))
    print(f"\nVerification: PyTorch vs ONNX Runtime")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Match: {'YES' if max_diff < 1e-4 else 'NO — check export!'}")


def main():
    parser = argparse.ArgumentParser(description='Export CLIP-ReID to ONNX')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--output', type=str, default='model.onnx')
    parser.add_argument('--num-classes', type=int, default=751)
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 128])
    args = parser.parse_args()

    img_size = tuple(args.img_size)
    cfg = get_cfg(img_size=img_size, num_classes=args.num_classes)

    print("Creating CLIP-ReID model...")
    model = make_model(cfg, num_class=args.num_classes, camera_num=0, view_num=0)

    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights: {args.weights}")
        model.load_param(args.weights)
    else:
        print("No weights — exporting random init model (for pipeline testing)")

    model.eval()
    export_to_onnx(model, args.output, img_size=img_size, device='cpu')
    verify_onnx_matches_pytorch(model, args.output, img_size=img_size, device='cpu')


if __name__ == '__main__':
    main()
