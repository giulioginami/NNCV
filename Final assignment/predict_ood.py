"""
OOD-aware prediction pipeline using MC Dropout.

For every test image the model runs N_MC_PASSES stochastic forward passes
(dropout kept active via MCDropout), averages the class probabilities, and
computes a predictive-entropy image score U_image.

- If U_image > tau  → image is flagged OOD  (include=False in CSV)
- If U_image <= tau → image is accepted as ID (include=True  in CSV)

In both cases the best-guess segmentation mask is saved so the server never
encounters a missing output file.  The CSV is what the server uses to decide
whether to score the image.

tau is loaded from /app/tau.txt, which is written by train.py whenever a new
best model checkpoint is saved (95th percentile of validation uncertainty).
"""
from pathlib import Path

import torch
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
import os
import csv

from model import Model

# Fixed paths inside participant container — do NOT change.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
TAU_PATH = "/app/tau.txt"   # OOD threshold saved during training

N_MC_PASSES = 30            # Number of stochastic forward passes


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(256, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return transform(img).unsqueeze(0)  # (1, C, H, W)


def postprocess(mean_probs: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """Convert mean class probabilities to a resized label map."""
    pred_max = torch.argmax(mean_probs, dim=1, keepdim=True)  # (1, 1, H, W)
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)
    return prediction.cpu().numpy().squeeze().astype(np.uint8)


def compute_uncertainty(mean_probs: torch.Tensor) -> float:
    """Predictive entropy averaged over all pixels → scalar image score."""
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)  # (1, H, W)
    return entropy.mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model — MCDropout stays active even in eval() mode.
    model = Model()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # Load the OOD threshold calibrated on the validation set.
    tau = float(open(TAU_PATH).read().strip())
    print(f"OOD threshold tau = {tau:.6f}")

    image_files = list(Path(IMAGE_DIR).glob("**/*.png"))
    print(f"Found {len(image_files)} images to process.")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = Path(OUTPUT_DIR) / "predictions.csv"
    predictions = []

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)

            # --- MC Dropout: N_MC_PASSES stochastic forward passes ---
            # Probabilities are accumulated in-place to keep memory usage low.
            mean_probs = None
            for _ in range(N_MC_PASSES):
                probs = model(img_tensor).softmax(dim=1)
                mean_probs = probs if mean_probs is None else mean_probs + probs
            mean_probs = mean_probs / N_MC_PASSES  # (1, C, H, W)

            # --- Gatekeeper ---
            u_image = compute_uncertainty(mean_probs)
            include = bool(u_image <= tau)

            if not include:
                print(f"[OOD ] {img_path.name}  U={u_image:.5f} > tau={tau:.5f}")

            # Always save a mask (avoids missing-file errors on the server).
            # The CSV include flag is what the server uses to exclude OOD images
            # from the Dice score.
            seg_pred = postprocess(mean_probs, original_shape)

            relative_path = img_path.relative_to(IMAGE_DIR)
            out_path = Path(OUTPUT_DIR) / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred).save(out_path)

            predictions.append({
                'image_name': str(relative_path).replace('\\', '/'),
                'include': include,
            })

    # Write CSV with one row per image
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'include'])
        writer.writeheader()
        writer.writerows(predictions)

    n_accepted = sum(p['include'] for p in predictions)
    n_rejected = len(predictions) - n_accepted
    print(f"Saved {len(predictions)} predictions to {csv_path}")
    print(f"Accepted (ID): {n_accepted}  |  Rejected (OOD): {n_rejected}")


if __name__ == "__main__":
    main()
