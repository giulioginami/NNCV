"""
This script implements a training loop for the model. It is designed to be flexible,
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console.
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block.
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    ToImage,
    ToDtype,
    ColorJitter,
    GaussianBlur,
)

from model import Model

PATCH_SIZE = 256   # model input resolution; must match training crop size

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def dice_loss(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for multi-class segmentation.

    Args:
        pred:         (B, C, H, W) raw logits
        target:       (B, H, W)   ground-truth train IDs
        ignore_index: pixels with this label are excluded from the loss
        eps:          small constant for numerical stability
    """
    pred_soft = torch.softmax(pred, dim=1)          # (B, C, H, W) — probabilities

    # Build a boolean mask for valid pixels and clamp ignored ones to 0
    valid = (target != ignore_index)                # (B, H, W)
    target_clamped = target.clone()
    target_clamped[~valid] = 0

    # One-hot encode the target: (B, H, W) → (B, C, H, W)
    target_onehot = torch.zeros_like(pred_soft)
    target_onehot.scatter_(1, target_clamped.unsqueeze(1), 1.0)

    # Zero out ignored pixels in both tensors
    valid_4d = valid.unsqueeze(1).expand_as(pred_soft)
    pred_soft    = pred_soft    * valid_4d
    target_onehot = target_onehot * valid_4d

    # Per-class Dice score, then average across classes
    intersection = (pred_soft * target_onehot).sum(dim=(0, 2, 3))
    denominator  = pred_soft.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
    dice         = (2.0 * intersection + eps) / (denominator + eps)

    return 1.0 - dice.mean()


class RandomCropDataset(torch.utils.data.Dataset):
    """
    Wraps a Cityscapes dataset and returns one random square crop per access.

    Applying the crop here (inside __getitem__) guarantees that the DataLoader
    workers always produce small PATCH_SIZE × PATCH_SIZE tensors, so GPU memory
    usage is independent of the original image resolution.  The same (i, j)
    offset is used for both the image and the mask, keeping them aligned.
    """
    def __init__(self, base_dataset, crop_size: int = PATCH_SIZE):
        self.base      = base_dataset
        self.crop_size = crop_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]            # (3, H, W), (1, H, W)
        _, H, W = image.shape
        i = torch.randint(0, H - self.crop_size + 1, (1,)).item()
        j = torch.randint(0, W - self.crop_size + 1, (1,)).item()
        image = image[:, i:i+self.crop_size, j:j+self.crop_size]
        mask  = mask[:,  i:i+self.crop_size, j:j+self.crop_size]
        return image, mask


def sliding_window_inference(model, images: torch.Tensor, patch_size: int, device) -> torch.Tensor:
    """
    Run inference on a batch of full images by tiling into overlapping patches.

    Patches overlap by 50% (stride = patch_size // 2).  For each position the
    patch is extracted from all B images and forwarded together.  Softmax
    probabilities are accumulated into a full-resolution buffer; overlapping
    regions are averaged before the final argmax, eliminating seams at patch
    borders.

    Args:
        model:      trained model in eval mode
        images:     (B, 3, H, W) normalized float tensors on CPU
        patch_size: square patch side length (must match training crop size)
        device:     inference device

    Returns:
        (B, H, W) long tensor of predicted class IDs on CPU
    """
    B, _, H, W = images.shape
    stride    = patch_size // 2
    n_classes = 19

    # Softmax accumulator and per-pixel coverage count (kept on CPU)
    prob_sum = torch.zeros(B, n_classes, H, W, dtype=torch.float32)
    count    = torch.zeros(B, 1,         H, W, dtype=torch.float32)

    row_starts = sorted(set(list(range(0, H - patch_size + 1, stride)) + [H - patch_size]))
    col_starts = sorted(set(list(range(0, W - patch_size + 1, stride)) + [W - patch_size]))

    for r in row_starts:
        for c in col_starts:
            patch  = images[:, :, r:r+patch_size, c:c+patch_size].to(device)  # (B, 3, P, P)
            logits = model(patch)                                               # (B, 19, P, P)
            probs  = torch.softmax(logits, dim=1).cpu()                        # (B, 19, P, P)
            prob_sum[:, :, r:r+patch_size, c:c+patch_size] += probs
            count[:,    0, r:r+patch_size, c:c+patch_size] += 1.0

    # Average overlapping regions then take argmax
    pred_masks = (prob_sum / count).argmax(dim=1).long()  # (B, H, W)
    return pred_masks


def hard_dice_score(pred_mask: torch.Tensor, target_mask: torch.Tensor,
                    n_classes: int = 19, ignore_index: int = 255,
                    eps: float = 1e-6) -> float:
    """
    Compute the mean Dice score from hard (argmax) predictions.

    Classes absent from both the prediction and the ground truth are skipped
    so they do not inflate the average.

    Args:
        pred_mask:   (H, W) long tensor — predicted class IDs
        target_mask: (H, W) long tensor — ground-truth train IDs

    Returns:
        float in [0, 1]; higher is better
    """
    valid = target_mask != ignore_index
    dices = []
    for c in range(n_classes):
        pred_c   = (pred_mask   == c) & valid
        target_c = (target_mask == c) & valid
        denom = pred_c.sum() + target_c.sum()
        if denom == 0:
            continue
        dices.append((2.0 * (pred_c & target_c).sum().float() + eps) / (denom.float() + eps))
    return torch.stack(dices).mean().item() if dices else 1.0


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the decoder and segmentation head")
    parser.add_argument("--encoder-lr", type=float, default=0.0001, help="Learning rate for the pretrained encoder")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random),
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training image transform: convert to float [0, 1] only.
    # No resize — images stay at original Cityscapes resolution (2048x1024).
    # No normalize — applied manually in the loop after colour augmentation.
    # The random crop to PATCH_SIZE happens inside RandomCropDataset.
    train_img_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
    ])

    # Validation image transform: full normalization pipeline, no resize.
    # Full-resolution images are needed for sliding-window evaluation.
    val_img_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Target (mask) transform — shared between train and val.
    # No resize; no scaling (class IDs must stay as integers).
    target_transform = Compose([
        ToImage(),
        ToDtype(torch.int64),
    ])

    # Transforms applied manually in the training loop
    img_normalize   = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_color_jitter  = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    train_gaussian_blur = GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    # Load the dataset and make a split for training and validation
    train_dataset = RandomCropDataset(
        Cityscapes(
            args.data_dir,
            split="train",
            mode="fine",
            target_type="semantic",
            transform=train_img_transform,
            target_transform=target_transform,
        ),
        crop_size=PATCH_SIZE,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=val_img_transform,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # batch_size=4 so four full images are processed per sliding-window call,
    # reducing GPU overhead during validation.
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer with separate learning rates for encoder and decoder.
    # The encoder is pretrained on ImageNet, so it needs a smaller lr to preserve
    # the learned features. The decoder and segmentation head are trained from
    # scratch, so they can use a larger lr.
    optimizer = AdamW([
        {'params': model.net.encoder.parameters(),            'lr': args.encoder_lr},
        {'params': model.net.decoder.parameters(),            'lr': args.lr},
        {'params': model.net.segmentation_head.parameters(),  'lr': args.lr},
    ])

    # Cosine annealing: smoothly decays each param group from its initial lr
    # down to eta_min over all epochs, reducing late-training oscillations.
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # (B, H, W)

            # Spatial augmentation: horizontal flip
            # Applied per sample so each image gets an independent flip decision.
            # The exact same flip is applied to the image and its mask to keep
            # them aligned. Color/blur augmentations must NOT touch the mask.
            flip_mask = torch.rand(images.shape[0], device=device) < 0.5
            images[flip_mask] = torch.flip(images[flip_mask], dims=[-1])
            labels[flip_mask] = torch.flip(labels[flip_mask], dims=[-1])

            # Image-only augmentations (applied before normalization)
            images = train_color_jitter(images)
            images = train_gaussian_blur(images)

            # Normalize after augmentation
            images = img_normalize(images)

            optimizer.zero_grad()
            outputs = model(images)

            # Combined loss: CrossEntropy + Dice
            ce   = criterion(outputs, labels)
            dice = dice_loss(outputs, labels)
            loss = ce + dice

            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss":      loss.item(),
                "train_ce_loss":   ce.item(),
                "train_dice_loss": dice.item(),
                "learning_rate":   optimizer.param_groups[0]['lr'],
                "epoch":           epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # Validation — sliding window over full 2048x1024 images
        model.eval()
        with torch.no_grad():
            dice_scores = []
            log_image   = True   # log one prediction image to wandb per epoch

            for i, (images, labels) in enumerate(valid_dataloader):
                # images: (B, 3, H, W)  labels: (B, 1, H, W)  — full resolution
                labels = convert_to_train_id(labels)
                labels = labels.long().squeeze(1)       # (B, H, W) on CPU

                # Stitch patch predictions for all B images in one call
                pred_masks = sliding_window_inference(
                    model, images, PATCH_SIZE, device
                )  # (B, H, W) long, CPU

                for b in range(images.shape[0]):
                    dice_scores.append(hard_dice_score(pred_masks[b], labels[b]))

                if log_image:
                    # Colorise the top-left 512x512 region of the first image
                    pred_vis = pred_masks[0, :512, :512].unsqueeze(0).unsqueeze(0)
                    gt_vis   = labels[0, :512, :512].unsqueeze(0).unsqueeze(0)
                    pred_img = convert_train_id_to_color(pred_vis)[0].permute(1, 2, 0).numpy()
                    gt_img   = convert_train_id_to_color(gt_vis)[0].permute(1, 2, 0).numpy()
                    wandb.log({
                        "predictions": [wandb.Image(pred_img)],
                        "labels":      [wandb.Image(gt_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
                    log_image = False

            valid_dice = sum(dice_scores) / len(dice_scores)
            valid_loss = 1.0 - valid_dice   # lower is better, consistent with training
            wandb.log({
                "valid_loss":      valid_loss,
                "valid_dice":      valid_dice,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)

        # scheduler.step()

    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
