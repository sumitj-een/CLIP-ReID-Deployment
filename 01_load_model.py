"""
Step 1: Load a .pth model and run PyTorch inference
====================================================

What this teaches:
    - How .pth files work (they're just saved weights)
    - How to load weights into a model architecture
    - How to preprocess images for the model
    - How to extract feature embeddings

CLIP-ReID takes a person image (256x128) and outputs a feature
embedding vector. Two images of the same person will have similar
embeddings. Different people will have different embeddings.

    Image → Model → Embedding (1280-dim vector)

    Compare embeddings:
        same person  → distance is small
        diff person  → distance is large

Usage:
    python 01_load_model.py --weights path/to/model.pth --image path/to/person.jpg

    # If no image provided, uses a random tensor for testing
    python 01_load_model.py --weights path/to/model.pth
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys


# ─── Step 1: Define the model architecture ───────────────────
# The .pth file only contains WEIGHTS (numbers).
# You need the architecture (structure) separately.
# Think of it like: .pth = the filling, architecture = the mold.
#
# For CLIP-ReID, the backbone is a Vision Transformer (ViT-B/16).
# We'll use a simplified version that extracts embeddings.

class SimpleViTReID(nn.Module):
    """
    Simplified ReID model using ViT backbone.
    In production, you'd use the full CLIP-ReID architecture.
    This demonstrates the concept: image → embedding.
    """
    def __init__(self, num_classes=751, embed_dim=768):
        super().__init__()
        # Load pretrained ViT-B/16 as backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dino:main', 'dino_vitb16', pretrained=True
        )
        self.embed_dim = embed_dim

        # Bottleneck: reduces dimension + normalizes
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Classifier head (used during training, optional during inference)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Extract features from ViT backbone
        features = self.backbone(x)  # (batch, 768)

        # Bottleneck projection
        projected = self.bottleneck(features)  # (batch, 512)

        if self.training:
            logits = self.classifier(projected)
            return logits, features, projected
        else:
            # During inference, return concatenated embedding
            embedding = torch.cat([features, projected], dim=1)  # (batch, 1280)
            # L2 normalize for cosine similarity
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
            return embedding


# ─── Step 2: Image preprocessing ─────────────────────────────
# The model expects a specific input format.
# This MUST match what the model was trained with.

def get_transform():
    """Preprocessing pipeline matching CLIP-ReID training."""
    return transforms.Compose([
        transforms.Resize((256, 128)),          # ReID standard: height > width
        transforms.ToTensor(),                   # PIL Image → tensor (0-1)
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],               # CLIP normalization
            std=[0.5, 0.5, 0.5]
        ),
    ])


def load_image(image_path, transform):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img)
    # Add batch dimension: (3, 256, 128) → (1, 3, 256, 128)
    return tensor.unsqueeze(0)


# ─── Step 3: Load model and run inference ─────────────────────

def main():
    parser = argparse.ArgumentParser(description='CLIP-ReID PyTorch Inference')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to .pth weights file')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to person image')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')
    args = parser.parse_args()

    # Select device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Create model ──
    print("Creating model architecture...")
    model = SimpleViTReID(num_classes=751, embed_dim=768)

    # ── Load weights (if provided) ──
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from: {args.weights}")
        # torch.load reads the .pth file
        # map_location ensures it works on CPU even if saved on GPU
        state_dict = torch.load(args.weights, map_location=device)

        # Some .pth files wrap weights in a dict with a key like 'model' or 'state_dict'
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # strict=False ignores mismatched keys (useful for partial loading)
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully!")
    else:
        print("No weights provided — using randomly initialized model (for demo)")

    model = model.to(device)
    model.eval()  # Set to inference mode (disables dropout, batchnorm in eval mode)

    # ── Prepare input ──
    transform = get_transform()

    if args.image and os.path.exists(args.image):
        print(f"Loading image: {args.image}")
        input_tensor = load_image(args.image, transform).to(device)
    else:
        print("No image provided — using random tensor (shape: 1, 3, 256, 128)")
        input_tensor = torch.randn(1, 3, 256, 128).to(device)

    # ── Run inference ──
    print("\nRunning inference...")
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        embedding = model(input_tensor)

    print(f"\n{'='*50}")
    print(f"Input shape:     {list(input_tensor.shape)}")
    print(f"Embedding shape: {list(embedding.shape)}")
    print(f"Embedding (first 10 values): {embedding[0][:10].cpu().numpy().round(4)}")
    print(f"Embedding norm:  {torch.norm(embedding[0]).item():.4f} (should be ~1.0 if L2 normalized)")

    # ── Compare two images (if running in production) ──
    print(f"\n{'='*50}")
    print("How ReID works:")
    print("  1. Extract embedding for person in Camera A")
    print("  2. Extract embedding for person in Camera B")
    print("  3. Compute cosine similarity between embeddings")
    print("  4. High similarity = same person")

    # Demo: compare with a second random input
    input2 = torch.randn(1, 3, 256, 128).to(device)
    with torch.no_grad():
        embedding2 = model(input2)

    similarity = torch.nn.functional.cosine_similarity(embedding, embedding2)
    print(f"\n  Similarity between two random inputs: {similarity.item():.4f}")
    print(f"  (Random inputs → low similarity, same person → high similarity)")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
