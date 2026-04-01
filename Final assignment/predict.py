"""
Prediction pipeline for the U-Net + ResNet-34 segmentation model.

Inference uses a sliding-window strategy that mirrors validation in train.py:
  1. The full image is normalized without resizing.
  2. It is tiled into non-overlapping PATCH_SIZE x PATCH_SIZE patches.
  3. All patches are forwarded through the model in one batched call.
  4. Predictions are stitched back into a full-resolution mask.

This keeps the model always operating at the resolution it was trained on
(PATCH_SIZE x PATCH_SIZE) while producing output at the original image size.
"""
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    Normalize,
)

from model import Model

# Fixed paths inside participant container — do NOT change.
IMAGE_DIR  = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

PATCH_SIZE = 256   # must match the crop size used during training


def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a normalized float tensor without resizing.

    Returns:
        (3, H, W) float32 tensor, ImageNet-normalized
    """
    transform = Compose([
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(img)   # (3, H, W) — no batch dimension


def predict_sliding_window(model, image_tensors: torch.Tensor, device) -> np.ndarray:
    """
    Run sliding-window inference on a batch of full images.

    For each patch position, the same patch is extracted from all B images and
    forwarded together, halving GPU overhead compared to one image at a time.

    Args:
        model:         trained model in eval mode
        image_tensors: (B, 3, H, W) normalized float tensors on CPU
        device:        inference device

    Returns:
        (B, H, W) uint8 numpy array of predicted class IDs
    """
    B, _, H, W = image_tensors.shape
    pred_masks = torch.zeros(B, H, W, dtype=torch.long)

    row_starts = sorted(set(list(range(0, H - PATCH_SIZE + 1, PATCH_SIZE)) + [H - PATCH_SIZE]))
    col_starts = sorted(set(list(range(0, W - PATCH_SIZE + 1, PATCH_SIZE)) + [W - PATCH_SIZE]))

    patches, positions = [], []
    for r in row_starts:
        for c in col_starts:
            patches.append(image_tensors[:, :, r:r+PATCH_SIZE, c:c+PATCH_SIZE])  # (B, 3, P, P)
            positions.append((r, c))

    batch  = torch.cat(patches, dim=0).to(device)    # (N*B, 3, P, P)
    logits = model(batch)                            # (N*B, 19, P, P)
    preds  = logits.argmax(dim=1).cpu()              # (N*B, P, P)

    for i, (r, c) in enumerate(positions):
        pred_masks[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE] = preds[i*B:(i+1)*B]

    return pred_masks.numpy().astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))  # DO NOT CHANGE
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for i in range(0, len(image_files), 4):
            batch_paths = image_files[i:i+4]
            tensors     = torch.stack([preprocess(Image.open(p).convert("RGB"))
                                       for p in batch_paths])          # (B, 3, H, W)
            seg_preds   = predict_sliding_window(model, tensors, device)  # (B, H, W)

            for j, img_path in enumerate(batch_paths):
                out_path = Path(OUTPUT_DIR) / img_path.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(seg_preds[j]).save(out_path)


if __name__ == "__main__":
    main()
