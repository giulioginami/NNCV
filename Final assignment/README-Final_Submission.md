# Neural Networks for Computer Vision — Final Assignment

**Student:** Giulio Ginami  
**Student number:** 2388715  
**TU/e email:** g.ginami@student.tue.nl  
**Course:** 5LSM0 — Neural Networks for Computer Vision, TU/e Q3

---

## Overview

This repository contains the full pipeline for semantic segmentation and out-of-distribution (OOD) detection on the Cityscapes dataset. The project is structured in three stages:

1. **Baseline** — vanilla U-Net trained at 256×256 with softmax confidence as OOD metric.
2. **Peak Performance** — ResNet-34 encoder U-Net with multi-scale inference (non-overlapping then overlapping sliding windows).
3. **OOD Detection** — Monte Carlo Dropout applied to the baseline and top-performance model; predictive entropy as a calibrated uncertainty metric.

---

## Installation

### Requirements

```
Python >= 3.9
torch >= 2.0
torchvision >= 0.15
segmentation-models-pytorch >= 0.3
numpy
Pillow
matplotlib
scikit-learn
wandb
```

### Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install segmentation-models-pytorch numpy Pillow matplotlib scikit-learn wandb
```

---

## Repository Structure

```
Final assignment/
│
├── Calibration datasets
│   ├── ID_dataset/              In-distribution images (Cityscapes val subset)
│   └── OOD_dataset/             Out-of-distribution images (sourced from public Kaggle datasets)
│
├── Model architectures
│   ├── model_baseline.py        Vanilla U-Net
│   ├── model_baseline_ood.py    Vanilla U-Net with MC Dropout
│   ├── model_peak.py            ResNet-34 U-Net (no dropout)
│   └── model.py                 ResNet-34 U-Net with MC Dropout
│
├── Trained weights
│   ├── model_baseline.pt        Trained vanilla U-Net
│   ├── model_baseline_ood.pt    Trained vanilla U-Net with MC Dropout
│   ├── model_no_overlap.pt      ResNet U-Net — non-overlapping multi-scale
│   ├── model_overlap.pt         ResNet U-Net — overlapping multi-scale
│   └── model_overlap_ood.pt     ResNet U-Net — overlapping multi-scale + MC Dropout
│
├── Training scripts
│   ├── train_baseline.py        Train the baseline U-Net
│   └── train.py                 Train the ResNet multi-scale models
│
├── Submission scripts
│   ├── predict_ood.py           OOD submission — baseline MC Dropout model
│   └── predict.py               OOD submission — ResNet multi-scale MC Dropout
│
├── Analysis
│   └── final_assignment.ipynb   Full results notebook (all sections)
│
└── Dockerfile                   Container for challenge server submission
```

---

## File Descriptions

### Calibration Datasets

| Folder | Description |
|---|---|
| `ID_dataset/` | 100 Cityscapes validation images used as the in-distribution reference for calibrating the entropy threshold |
| `OOD_dataset/` | 99 out-of-distribution images from external sources, used alongside `ID_dataset/` to set the OOD detection threshold |

Both folders are consumed exclusively by `final_assignment.ipynb`.

---

### Model Architectures

#### `model_baseline.py`
Vanilla U-Net with 4 encoder levels and symmetric decoder with skip connections. No dropout. Trained at 256×256 with mean/std normalization of 0.5.

#### `model_baseline_ood.py`
Same vanilla U-Net as above, with `Dropout2d(p=0.5)` integrated into every `DoubleConv` block — covering encoder, bottleneck, and all four decoder levels (9 Dropout2d layers total). Dropout acts as regularisation during training and is re-enabled at inference via `enable_dropout(model)` for MC Dropout uncertainty estimation.

#### `model_peak.py`
U-Net with a pretrained ResNet-34 encoder (`encoder_weights='imagenet'`) via `segmentation_models_pytorch`. No dropout. Trained on 256×256 random crops with ImageNet normalization.

#### `model.py`
Same ResNet-34 U-Net as `model_peak.py` with `Dropout2d(p=0.5)` appended after each of the 5 decoder block convolutions. Exposes `enable_dropout(model)` for MC Dropout inference.

---

### Training Scripts

#### `train_baseline.py` — Baseline U-Net

| Setting | Value |
|---|---|
| Input resolution | 256×256 (resize) |
| Normalization | mean = std = 0.5 |
| Loss | Cross-Entropy |
| Optimizer | AdamW |

#### `train.py` — ResNet Multi-Scale Models

Trains `model.py` (ResNet-34 U-Net with MC Dropout) with the following enhancements:

| Setting | Value |
|---|---|
| Input | Random 256×256 crops from 2048×1024 images |
| Normalization | ImageNet statistics |
| Augmentation | `ColorJitter`, `GaussianBlur` |
| Loss | Combined Cross-Entropy + Dice |
| Optimizer | AdamW with differential learning rates (lower LR for pretrained encoder) |

---

### Submission Scripts

#### `predict_ood.py` — Baseline MC Dropout Submission

Used with `model_baseline_ood.pt` and `model_baseline_ood.py` for the **OOD challenge**.

**Pipeline per image:**
1. Resize to 256×256, normalize with mean/std = 0.5
2. Run 30 stochastic forward passes (Dropout active)
3. Average softmax outputs → mean prediction → per-pixel predictive entropy
4. If `mean_entropy >= ENTROPY_THRESHOLD` → `include = False` (OOD)
5. Save segmentation mask for every image
6. Write `predictions.csv` with `image_name` and `include` flag

> Set `ENTROPY_THRESHOLD` in the script to the value produced by **Section 3.1** of `final_assignment.ipynb`.


#### `predict.py` — Multi-Scale MC Dropout Submission

Used with `model_overlap_ood.pt` and `model.py` for the **OOD challenge**.

**Pipeline per image:**
1. Resize to 2048×1024, apply ImageNet normalization
2. **MC uncertainty pass** (Dropout ON): 30 lo-res stochastic passes at 128×256 → predictive entropy → OOD flag
3. **Hi-res sliding-window pass** (Dropout OFF): overlapping 256×256 patches with stride 128 (50% overlap) → averaged patch probabilities
4. **Class-specific blend**: per-class weighting of hi-res and lo-res probabilities — local classes weighted higher hi-res detail; global classes weighted higher lo-res context
5. ID images → blended segmentation

> Set `ENTROPY_THRESHOLD` in the script to the value produced by **Section 3.2** of `final_assignment.ipynb`.

---

### Analysis Notebook

#### `final_assignment.ipynb`

Notebook for analysis of results and threshold calibration. Run all cells top-to-bottom. Requires `ID_dataset/`, `OOD_dataset/`, and all five `.pt` checkpoint files in the same directory.

| Section | Checkpoint | Description |
|---|---|---|
| 1 — Baseline | `model_baseline.pt` | Segmentation on ID images; softmax entropy as OOD metric; entropy and max-confidence histograms |
| 2.1 — Non-overlapping | `model_no_overlap.pt` | Multi-scale inference with non-overlapping 256×256 patches + lo-res pass; fixed blend weight 0.8/0.2 |
| 2.2 — Overlapping | `model_overlap.pt` | Multi-scale inference with 50% overlapping patches + lo-res pass; class-specific blend weights |
| 3.1 — MC Dropout Baseline | `model_baseline_ood.pt` | MC Dropout (T=30) on baseline U-Net; predictive entropy histograms; OOD threshold calibration |
| 3.2 — MC Dropout ResNet | `model_overlap_ood.pt` | MC Dropout at 128×256 for entropy; full multi-scale segmentation for ID images; OOD threshold calibration |

---

### Trained Model Weights

| File | Architecture | Used in |
|---|---|---|
| `model_baseline.pt` | Vanilla U-Net | Section 1, `predict_ood.py` baseline |
| `model_baseline_ood.pt` | Vanilla U-Net + MC Dropout | Section 3.1, `predict_ood.py` |
| `model_no_overlap.pt` | ResNet-34 U-Net | Section 2.1 |
| `model_overlap.pt` | ResNet-34 U-Net | Section 2.2 |
| `model_overlap_ood.pt` | ResNet-34 U-Net + MC Dropout | Section 3.2, `predict.py` |

---

## Challenge Server Endpoints

> Accessible only from the TU/e campus network or VPN.

| Benchmark | URL |
|---|---|
| Baseline / Peak Performance | http://131.155.126.249:5001/ |
| Robustness | http://131.155.126.249:5002/ |
| Efficiency | http://131.155.126.249:5003/ |
| Out-of-Distribution | http://131.155.126.249:5004/ |
