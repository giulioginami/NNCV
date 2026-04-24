"""
Prediction pipeline — ResNet-34 U-Net, multi-scale inference, MC Dropout OOD detection.

Steps per image:
  1. Preprocess: resize to 2048x1024, ImageNet-normalize.
  2. MC uncertainty: T lo-res stochastic passes (128x256) → entropy → include flag.
  3. Hi-res sliding-window (Dropout OFF): overlapping 256x256 patches, stride=128.
  4. Class-specific blend of averaged MC lo-res probs and hi-res probs → mask.
  5. Save mask for every image; append include flag to predictions.csv.
"""
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

from model import Model, enable_dropout

# Fixed paths inside participant container — do NOT change.
IMAGE_DIR  = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

PATCH_SIZE = 256
PATCH_LO_H = 128
PATCH_LO_W = 256

MC_PASSES         = 30
ENTROPY_THRESHOLD = 1.6906

# Class-specific hi-res blending weights
_HI_WEIGHT_PER_CLASS = [
    0.70,  #  0  road
    0.70,  #  1  sidewalk
    0.70,  #  2  building
    0.80,  #  3  wall
    0.80,  #  4  fence
    1.00,  #  5  pole
    1.00,  #  6  traffic light
    1.00,  #  7  traffic sign
    0.75,  #  8  vegetation
    0.80,  #  9  terrain
    1.00,  # 10  sky
    1.00,  # 11  person
    1.00,  # 12  rider
    0.90,  # 13  car
    0.90,  # 14  truck
    0.90,  # 15  bus
    0.90,  # 16  train
    1.00,  # 17  motorcycle
    1.00,  # 18  bicycle
]

CLASS_HI_WEIGHT = torch.tensor(_HI_WEIGHT_PER_CLASS, dtype=torch.float32).view(19, 1, 1)
CLASS_LO_WEIGHT = 1.0 - CLASS_HI_WEIGHT


def preprocess(img: Image.Image) -> torch.Tensor:
    """Resize to 2048x1024 and ImageNet-normalize. Returns (3, 1024, 2048) CPU tensor."""
    img = img.resize((2048, 1024), Image.BILINEAR)
    transform = Compose([
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(img)


def predict(model, img_tensor: torch.Tensor, device):
    """
    Full multi-scale inference + MC Dropout OOD detection for a single image.

    Args:
        model:      trained Model (eval mode, Dropout2d present)
        img_tensor: (3, 1024, 2048) CPU tensor from preprocess()
        device:     inference device

    Returns:
        seg_mask    : (H, W) uint8 numpy array — class IDs 0-18
        mean_entropy: float — scalar predictive entropy
        include     : bool  — True = ID, False = OOD
    """
    img_t = img_tensor.unsqueeze(0)          # (1, 3, H, W)
    _, _, H, W = img_t.shape

    # 1. MC lo-res passes (Dropout ON) → entropy → include flag 
    enable_dropout(model)
    lo_input = F.interpolate(
        img_t, size=(PATCH_LO_H, PATCH_LO_W), mode='bilinear', align_corners=False
    ).to(device)

    mc_probs_list = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            mc_probs_list.append(torch.softmax(model(lo_input), dim=1).cpu())

    mean_lo_probs = torch.stack(mc_probs_list).mean(dim=0)   # (1, 19, LH, LW)

    eps          = 1e-10
    entropy_map  = -(mean_lo_probs * torch.log(mean_lo_probs + eps)).sum(dim=1)
    mean_entropy = float(entropy_map.mean())
    include      = mean_entropy < ENTROPY_THRESHOLD

    # 2. Hi-res overlapping sliding window (Dropout OFF) 
    model.eval()
    prob_hi = torch.zeros(1, 19, H, W)
    count   = torch.zeros(1,  1, H, W)
    stride  = PATCH_SIZE // 2   # 128 — 50% overlap

    row_starts = sorted(set(list(range(0, H - PATCH_SIZE + 1, stride)) + [H - PATCH_SIZE]))
    col_starts = sorted(set(list(range(0, W - PATCH_SIZE + 1, stride)) + [W - PATCH_SIZE]))

    with torch.no_grad():
        for r in row_starts:
            for c in col_starts:
                patch = img_t[:, :, r:r+PATCH_SIZE, c:c+PATCH_SIZE].to(device)
                probs = torch.softmax(model(patch), dim=1).cpu()
                prob_hi[:, :, r:r+PATCH_SIZE, c:c+PATCH_SIZE] += probs
                count[:,  0, r:r+PATCH_SIZE, c:c+PATCH_SIZE]  += 1.0

    prob_hi /= count   # (1, 19, H, W)

    # 3. Upsample MC lo-res probs + class-specific blend 
    lo_full  = F.interpolate(mean_lo_probs, size=(H, W), mode='bilinear', align_corners=False)
    blended  = CLASS_HI_WEIGHT * prob_hi + CLASS_LO_WEIGHT * lo_full
    seg_mask = blended[0].argmax(dim=0).numpy().astype(np.uint8)   # (H, W)

    return seg_mask, mean_entropy, include


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("**/*.png"))
    print(f"Found {len(image_files)} images to process.")
    print(f"MC Dropout: T={MC_PASSES} passes, entropy threshold={ENTROPY_THRESHOLD:.4f}")
    print(f"Multi-scale: hi={PATCH_SIZE}x{PATCH_SIZE} (50% overlap), "
          f"lo={PATCH_LO_H}x{PATCH_LO_W}, class-specific blend")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path    = Path(OUTPUT_DIR) / "predictions.csv"
    predictions = []

    for img_path in image_files:
        img   = Image.open(img_path).convert("RGB")
        img_t = preprocess(img)

        seg_mask, mean_entropy, include = predict(model, img_t, device)

        relative_path = img_path.relative_to(IMAGE_DIR)
        out_path      = Path(OUTPUT_DIR) / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(seg_mask).save(out_path)

        predictions.append({
            'image_name': str(relative_path).replace('\\', '/'),
            'include':    include,
        })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'include'])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions and CSV to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
