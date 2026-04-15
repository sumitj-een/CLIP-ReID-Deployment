"""
Batch ReID test — loads model once, runs all comparisons in memory.
Reports same-person vs different-person score distribution.
"""

import sys
import os
import itertools
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, '/app/clip_reid_repo')
from config import cfg
from model.make_model_clipreid import make_model

# ── Config ──────────────────────────────────────────────────────────────────
WEIGHTS  = '/app/Market1501_clipreid_ViT-B-16_60.pth'
CONFIG   = '/app/clip_reid_repo/configs/person/vit_clipreid.yml'
PERSONS  = ["0002", "0007", "0010", "0011", "0012", "0020", "0022", "0023"]
IMG_DIR  = '/app'

# ── Load config ──────────────────────────────────────────────────────────────
cfg.merge_from_file(CONFIG)
cfg.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ── Load model once ──────────────────────────────────────────────────────────
print("Loading model...")
model = make_model(cfg, num_class=751, camera_num=0, view_num=0)
state_dict = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()
print("Model ready.\n")

# ── Image transform ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    return F.normalize(emb, dim=1)

# ── Precompute all embeddings ────────────────────────────────────────────────
print("Computing embeddings for all images...")
embeddings = {}
for pid in PERSONS:
    for suffix in ['a', 'b']:
        path = f"{IMG_DIR}/{pid}_{suffix}.jpg"
        if os.path.exists(path):
            embeddings[f"{pid}_{suffix}"] = get_embedding(path)
            print(f"  {pid}_{suffix} done")
print()

# ── Same-person pairs ────────────────────────────────────────────────────────
same_scores = []
print("=== SAME PERSON PAIRS (intra-class) ===")
for pid in PERSONS:
    key_a, key_b = f"{pid}_a", f"{pid}_b"
    if key_a in embeddings and key_b in embeddings:
        score = torch.cosine_similarity(embeddings[key_a], embeddings[key_b]).item()
        same_scores.append(score)
        print(f"  {pid} vs {pid}: {score:.4f}")

# ── Different-person pairs ───────────────────────────────────────────────────
diff_scores = []
print("\n=== DIFFERENT PERSON PAIRS (inter-class) ===")
for pid1, pid2 in itertools.combinations(PERSONS, 2):
    key1, key2 = f"{pid1}_a", f"{pid2}_a"
    if key1 in embeddings and key2 in embeddings:
        score = torch.cosine_similarity(embeddings[key1], embeddings[key2]).item()
        diff_scores.append(score)
        print(f"  {pid1} vs {pid2}: {score:.4f}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Same person  — min: {min(same_scores):.4f}  max: {max(same_scores):.4f}  mean: {sum(same_scores)/len(same_scores):.4f}")
print(f"Diff person  — min: {min(diff_scores):.4f}  max: {max(diff_scores):.4f}  mean: {sum(diff_scores)/len(diff_scores):.4f}")
suggested = (min(same_scores) + max(diff_scores)) / 2
print(f"\nSuggested threshold: {suggested:.4f}")
