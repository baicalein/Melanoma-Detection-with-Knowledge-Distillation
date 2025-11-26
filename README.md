# Melanoma Detection with Knowledge Distillation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project implements knowledge distillation for melanoma detection on the HAM10000 dataset, targeting deployment on mobile phones and edge devices.

## Table of Contents

- [Melanoma Detection with Knowledge Distillation](#melanoma-detection-with-knowledge-distillation)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
  - [Reproducing Results](#reproducing-results)
    - [Step 1: Train Teacher Model](#step-1-train-teacher-model)
    - [Step 2: Train Student with Knowledge Distillation](#step-2-train-student-with-knowledge-distillation)
    - [Step 3: Quantize Student Model](#step-3-quantize-student-model)
  - [Configuration](#configuration)
  - [Results](#results)
    - [Deployment Metrics](#deployment-metrics)
  - [Important Links](#important-links)
  - [Contact](#contact)
  - [Development Commands](#development-commands)
  - [Changelog](#changelog)

## Project Structure

```
├── src/
│   ├── config.py              # Centralized configuration
│   ├── data/
│   │   ├── dataset.py         # HAM10000 Dataset class
│   │   └── splits.py          # Lesion-aware stratified splitting
│   ├── models/
│   │   ├── architectures.py   # Teacher (ResNet) & Student (MobileNetV3)
│   │   └── kd_loss.py         # Knowledge distillation & focal loss
│   ├── training/
│   │   └── trainer.py         # Training loops with early stopping
│   ├── evaluation/
│   │   ├── metrics.py         # ROC-AUC, PR-AUC, ECE, calibration
│   │   └── quantization.py    # INT8 quantization utilities
│   └── plotting/
│       ├── eda.py             # EDA visualizations
│       └── training_plots.py  # Training curves, reliability diagrams
├── scripts/
│   ├── train_teacher.py       # Train teacher model
│   ├── train_student.py       # Train student with KD
│   └── quantize_model.py      # Quantize and evaluate
├── notebooks/
│   ├── 00_eda.ipynb           # Exploratory data analysis
│   └── 01_benchmarks.ipynb    # Model benchmarks
├── data/
│   ├── raw/ham_1000_archive/  # Raw HAM10000 images
│   └── processed/             # Processed splits (train/val/holdout)
├── models/
│   ├── checkpoints/           # Saved model weights
│   └── logs/                  # Training logs
└── artifacts/
    ├── imgs/                  # Generated figures
    └── tbls/                  # Generated tables
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd Deep_Learning_Final_Project

# Create virtual environment (using uv or pip)
uv venv && source .venv/bin/activate

# Install dependencies
make install
# or: pip install -e .

# Verify installation
python -c "from src import set_seed; print('OK')"
```

## Dataset Setup

1. Download HAM10000 from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or ISIC Archive
2. Extract to `data/raw/ham_1000_archive/`
3. Run data preprocessing:

```bash
# Create labeled CSV with lesion types and binary targets
python -c "from src.data.build_data import create_base_df; create_base_df()"

# Create lesion-aware train/val/holdout splits (70/15/15)
python -c "from src.data.splits import load_or_create_splits; load_or_create_splits(force_recreate=True)"
```

Expected output:

- `data/processed/labeled_ham10000.csv` - Full labeled dataset
- `data/processed/train_data.csv` - Training set (~7,012 images)
- `data/processed/val_data.csv` - Validation set (~1,502 images)
- `data/processed/holdout_data.csv` - Holdout/test set (~1,503 images)
- `data/processed/split_metadata.json` - Split statistics and seed

## Reproducing Results

### Step 1: Train Teacher Model

```bash
python scripts/train_teacher.py \
    --architecture resnet34 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --loss focal \
    --patience 10 \
    --seed 42
```

Expected outputs:

- `models/checkpoints/teacher_resnet34_focal_best.pth`
- `artifacts/imgs/training/teacher_resnet34_focal/` (training curves, reliability diagram)

### Step 2: Train Student with Knowledge Distillation

Run with recommended hyperparameters (T=2, α=0.5):

```bash
python scripts/train_student.py \
    --teacher-ckpt models/checkpoints/teacher_resnet34_focal_best.pth \
    --teacher-arch resnet34 \
    --student-arch mobilenet_v3_small \
    --temperature 2 \
    --alpha 0.5 \
    --epochs 50 \
    --seed 42
```

For ablation studies, also run:

- T=1, α=0.5
- T=2, α=0.9
- T=1, α=0.9

### Step 3: Quantize Student Model

```bash
python scripts/quantize_model.py \
    --model-ckpt models/checkpoints/student_T2_alpha0.5_best.pth \
    --method dynamic
```

Expected outputs:

- `models/checkpoints/quantized_dynamic_quantized.pth`
- `artifacts/imgs/quantization/` (deployment comparison plots)

## Configuration

All hyperparameters are centralized in `src/config.py`:

```python
from src.config import (
    DataConfig,      # batch_size, image_size, augmentation
    TeacherConfig,   # architecture, dropout, pretrained
    StudentConfig,   # architecture, size constraints
    TrainingConfig,  # epochs, lr, loss_type, early_stopping
    KDConfig,        # temperature, alpha (KD weight)
)
```

## Results

### Deployment Metrics

TODO: fill these in
TODO: consider having this auto update from pandas

| Model | Size (MB) | Latency (ms) | ROC-AUC | ECE |
|-------|-----------|--------------|---------|-----|
| Teacher (ResNet-34) | ~85 | ~15 | TBD | TBD |
| Student (MobileNetV3-Small) | ~10 | ~5 | TBD | TBD |
| Student (INT8 Quantized) | ~3 | ~3 | TBD | TBD |

## Important Links

- [Overleaf working document](https://www.overleaf.com/read/ycgbrjvyyqbx#5162eb)
- [TA Rivanna Guide](https://github.com/JustUnoptimized/ds6050-rivanna)

## Contact

Ryan Healy (rah5ff) and Angie Yoon

---

## Development Commands

```bash
make install       # Install dependencies
make test          # Run tests with coverage
make lint          # Run linters (ruff, mypy)
make format        # Format code (black, isort)
make clean         # Clean build artifacts
make check-all     # Run format, lint, and test
```

## Changelog

- **Lesion-aware data splitting**: Prevents data leakage by ensuring images from the same lesion don't appear in different splits
- **Teacher model**: ResNet-34 with focal loss for handling class imbalance (~11% melanoma prevalence)
- **Student model**: MobileNetV3-Small for mobile deployment (<25 MB)
- **Knowledge distillation**: Temperature-scaled KD with focused hyperparameter search (T ∈ {1, 2}, α ∈ {0.5, 0.9})
- **Comprehensive evaluation**: ROC-AUC, PR-AUC, sensitivity/specificity at 95% recall, ECE calibration
- **INT8 quantization**: Post-training quantization for edge deployment
