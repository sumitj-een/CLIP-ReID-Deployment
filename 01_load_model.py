"""
Step 1: Load a CLIP-ReID .pth model and run PyTorch inference
==============================================================

What this teaches:
    - How .pth files work (they're just saved weights)
    - How to load weights into the REAL CLIP-ReID architecture
    - How to preprocess images for the model
    - How to extract feature embeddings for person re-identification

CLIP-ReID takes a person image (256x128) and outputs a feature
embedding vector (1280-dim for ViT-B-16). Two images of the same
person will have similar embeddings.

    Image → Model → Embedding (1280-dim vector)

Usage:
    # With pretrained weights
    python 01_load_model.py --weights path/to/ViT-B-16_60.pth --image person.jpg

    # Without weights (random init, for testing the pipeline)
    python 01_load_model.py

    # Compare two images
    python 01_load_model.py --weights path/to/model.pth --image1 person_cam1.jpg --image2 person_cam2.jpg
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys

# ─── Add CLIP-ReID repo to Python path ───────────────────────
# This lets us import the real model code from the submodule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'clip_reid_repo'))

from config import cfg as default_cfg
from model.make_model_clipreid import make_model


# ─── Step 1: Build config for the model ──────────────────────
# CLIP-ReID uses a config system (YACS). We need to set the right
# values to match the .pth weights we're loading.

def get_cfg(config_file=None, img_size=(256, 128), model_name='ViT-B-16',
            num_classes=751, dataset='market1501'):
    """
    Create config for CLIP-ReID model.

    Args:
        config_file: path to .yml config (optional, uses defaults if None)
        img_size: (height, width) — must match training config
        model_name: 'ViT-B-16' or 'RN50'
        num_classes: number of person IDs in the training dataset
        dataset: dataset name (affects prompt text)
    """
    cfg = default_cfg.clone()

    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        # Set defaults for ViT-B-16 on Market-1501
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


# ─── Step 2: Image preprocessing ─────────────────────────────
# Must match the training preprocessing exactly.

def get_transform(img_size=(256, 128)):
    """Preprocessing pipeline matching CLIP-ReID training."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],    # CLIP normalization
            std=[0.5, 0.5, 0.5]
        ),
    ])


def load_image(image_path, transform):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img)
    return tensor.unsqueeze(0)  # add batch dim: (3,H,W) → (1,3,H,W)


# ─── Step 3: Main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CLIP-ReID Inference')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to .pth weights file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to .yml config file')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image for embedding extraction')
    parser.add_argument('--image1', type=str, default=None,
                        help='First image for ReID comparison')
    parser.add_argument('--image2', type=str, default=None,
                        help='Second image for ReID comparison')
    parser.add_argument('--num-classes', type=int, default=751,
                        help='Number of classes (751 for Market-1501)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 128],
                        help='Image size: height width')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')
    args = parser.parse_args()

    # Select device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Build config ──
    img_size = tuple(args.img_size)
    cfg = get_cfg(
        config_file=args.config,
        img_size=img_size,
        num_classes=args.num_classes,
    )
    print(f"Model: {cfg.MODEL.NAME}")
    print(f"Image size: {img_size}")
    print(f"Num classes: {args.num_classes}")

    # ── Create model ──
    # make_model needs: cfg, num_class, camera_num, view_num
    # For inference without SIE, camera_num and view_num don't matter
    print("Creating CLIP-ReID model...")
    model = make_model(cfg, num_class=args.num_classes, camera_num=0, view_num=0)

    # ── Load weights ──
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights: {args.weights}")
        model.load_param(args.weights)
    else:
        print("No weights provided — using random initialization (for pipeline testing)")

    model = model.to(device)
    model.eval()

    # ── Prepare transform ──
    transform = get_transform(img_size)

    # ── Run inference ──
    if args.image1 and args.image2:
        # Compare two images
        print(f"\nComparing two images...")
        img1 = load_image(args.image1, transform).to(device)
        img2 = load_image(args.image2, transform).to(device)

        with torch.no_grad():
            emb1 = model(img1)
            emb2 = model(img2)

        # Normalize
        emb1 = nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = nn.functional.normalize(emb2, p=2, dim=1)

        similarity = torch.cosine_similarity(emb1, emb2).item()

        print(f"  Image 1: {args.image1}")
        print(f"  Image 2: {args.image2}")
        print(f"  Embedding shape: {list(emb1.shape)}")
        print(f"  Cosine similarity: {similarity:.4f}")
        print(f"  Match: {'LIKELY SAME PERSON' if similarity > 0.5 else 'DIFFERENT PEOPLE'}")

    else:
        # Single image (or random input)
        if args.image and os.path.exists(args.image):
            print(f"\nLoading image: {args.image}")
            input_tensor = load_image(args.image, transform).to(device)
        else:
            print("\nNo image — using random tensor")
            input_tensor = torch.randn(1, 3, *img_size).to(device)

        with torch.no_grad():
            embedding = model(input_tensor)

        # Normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)

        print(f"\n{'='*50}")
        print(f"Input shape:     {list(input_tensor.shape)}")
        print(f"Embedding shape: {list(embedding.shape)}")
        print(f"Embedding (first 10): {embedding[0][:10].cpu().numpy().round(4)}")
        print(f"Embedding norm:  {torch.norm(embedding[0]).item():.4f}")
        print(f"{'='*50}")


if __name__ == '__main__':
    main()
