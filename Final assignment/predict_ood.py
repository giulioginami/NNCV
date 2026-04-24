"""
Prediction pipeline for the baseline U-Net with Monte Carlo Dropout.

For each image:
  1. Preprocess: resize to 256x256, normalize with (0.5, 0.5, 0.5).
  2. Run T=MC_PASSES forward passes with dropout active (MC Dropout).
  3. Average the T softmax outputs → mean prediction (1, 19, H, W).
  4. Segmentation mask: argmax of the mean prediction, resized to original shape.
  5. Predictive entropy: per-pixel entropy of the mean prediction → scalar mean.
  6. OOD decision: mean entropy >= ENTROPY_THRESHOLD → include=False (OOD).
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from model import Model, enable_dropout
import os
import csv

# Fixed paths inside participant container — do NOT change.
IMAGE_DIR  = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

# MC Dropout settings
MC_PASSES = 30   # number of stochastic forward passes per image

# Predictive-entropy threshold for OOD detection.
# Images with mean entropy >= threshold → include=False (OOD).
ENTROPY_THRESHOLD = 1.2272  


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return transform(img).unsqueeze(0)   # (1, 3, 256, 256)


def mc_predict(model: nn.Module, img_tensor: torch.Tensor):
    """
    Run MC_PASSES stochastic forward passes and return:
      mean_pred  : (1, 19, 256, 256) — average softmax over all passes
      entropy    : float             — mean per-pixel predictive entropy (nats)

    Dropout must already be enabled on the model before calling this function.
    """
    passes = []
    for _ in range(MC_PASSES):
        logits = model(img_tensor)                         # (1, 19, 256, 256)
        passes.append(torch.softmax(logits, dim=1))        # (1, 19, 256, 256)

    # Mean prediction: average softmax across all T passes
    mean_pred = torch.stack(passes, dim=0).mean(dim=0)    # (1, 19, 256, 256)

    # Predictive entropy: H(mean distribution) per pixel → scalar mean
    eps         = 1e-10
    entropy_map = -(mean_pred * torch.log(mean_pred + eps)).sum(dim=1)  # (1, 256, 256)
    mean_entropy = entropy_map.mean().item()

    return mean_pred, mean_entropy


def postprocess(mean_pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """
    Argmax of the mean prediction, resized back to the original image shape.

    Args:
        mean_pred      : (1, 19, 256, 256) averaged softmax tensor
        original_shape : (H, W) of the original PIL image

    Returns:
        (H, W) uint8 numpy array of predicted class IDs
    """
    pred_max   = mean_pred.argmax(dim=1, keepdim=True)            # (1, 1, 256, 256)
    prediction = Resize(size=original_shape,
                        interpolation=InterpolationMode.NEAREST)(pred_max)
    return prediction.cpu().numpy().squeeze().astype(np.uint8)    # (H, W)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and activate MC Dropout
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    enable_dropout(model)   # keep Dropout2d layers active at inference time
    model.to(device)

    image_files = list(Path(IMAGE_DIR).glob("**/*.png"))
    print(f"Found {len(image_files)} images to process.")
    print(f"MC Dropout: T={MC_PASSES} passes, entropy threshold={ENTROPY_THRESHOLD}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path    = Path(OUTPUT_DIR) / "predictions.csv"
    predictions = []

    with torch.no_grad():
        for img_path in image_files:
            img            = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)

            # T stochastic forward passes → mean prediction + entropy
            mean_pred, mean_entropy = mc_predict(model, img_tensor)

            # Segmentation mask from mean prediction
            seg_mask = postprocess(mean_pred, original_shape)

            # OOD decision
            include_decision = mean_entropy < ENTROPY_THRESHOLD

            # Save mask
            relative_path = img_path.relative_to(IMAGE_DIR)
            out_path      = Path(OUTPUT_DIR) / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_mask).save(out_path)

            predictions.append({
                'image_name': str(relative_path).replace('\\', '/'),
                'include':    bool(include_decision),
            })

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'include'])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {csv_path}")


if __name__ == "__main__":
    main()
