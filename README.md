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
      - [Complete Model Comparison](#complete-model-comparison)
      - [Teacher Model Comparison](#teacher-model-comparison)
      - [Knowledge Distillation Effectiveness](#knowledge-distillation-effectiveness)
      - [Latency Benchmarks](#latency-benchmarks)
      - [Holdout Set Evaluation](#holdout-set-evaluation)
      - [Teacher Threshold Curves](#teacher-threshold-curves)
      - [Student Threshold Curves](#student-threshold-curves)
      - [Training Curves (Student T=1, α=0.5)](#training-curves-student-t1-α05)
      - [ROC \& PR Curves](#roc--pr-curves)
      - [Reliability Diagram (Calibration)](#reliability-diagram-calibration)
      - [Teacher vs Student Comparison](#teacher-vs-student-comparison)
      - [Teacher vs Student Predictions](#teacher-vs-student-predictions)
      - [Confidence Distribution](#confidence-distribution)
      - [Challenging Cases](#challenging-cases)
      - [High Confidence Errors](#high-confidence-errors)
    - [Sklearn Baselines](#sklearn-baselines)
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

**Teacher Architecture Comparison:**

Train all teacher architectures at once (skips any that already have checkpoints):

```bash
# Train ALL teachers (ResNet + EfficientNet)
make train-teacher

# Or train specific families:
make train-resnet-teachers       # ResNet-18/34/50/101/152
make train-efficientnet-teachers # EfficientNet-B0 through B7

# Or train a single architecture:
make train-teacher-single TEACHER_ARCH=resnet50
```

Or use `make summary` to see which experiments are complete and get commands for missing ones.

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

| Model | Size (MB) | Latency (ms) | ROC-AUC | ECE |
|-------|-----------|--------------|---------|-----|
| ResNet-18 | 42.7 | 1.9 | 0.900 | 0.195 |
| ResNet-34 | 81.3 | 3.4 | 0.898 | 0.205 |
| ResNet-50 | 89.9 | 4.9 | 0.866 | 0.559 |
| ResNet-101 | 162.5 | 9.0 | 0.886 | 0.221 |
| ResNet-152 | 222.4 | 14.4 | 0.900 | 0.219 |
| EfficientNet-B0 | 15.5 | 6.0 | 0.904 | 0.157 |
| EfficientNet-B1 | 25.1 | 8.6 | **0.919** | 0.174 |
| EfficientNet-B2 | 29.6 | 8.9 | 0.904 | **0.064** |
| EfficientNet-B3 | 41.1 | 9.7 | 0.908 | 0.076 |
| EfficientNet-B4 | 67.4 | 11.9 | 0.906 | 0.115 |
| EfficientNet-B5 | 108.8 | 14.8 | 0.899 | 0.235 |
| EfficientNet-B6 | 156.3 | 17.2 | 0.890 | 0.145 |
| EfficientNet-B7 | 244.5 | 20.4 | 0.917 | 0.169 |
| Student (T=1, α=0.5) | ~9.1 | ~3 | **0.921** | 0.072 |
| Student (T=2, α=0.5) | ~9.1 | ~3 | 0.920 | 0.134 |

<details>
<summary><b>📊 Model Comparison Charts (click to expand)</b></summary>

#### Complete Model Comparison
![Complete Model Comparison](artifacts/imgs/01_baselines/complete_model_comparison.png)

#### Teacher Model Comparison
![Teacher Comparison](artifacts/imgs/01_baselines/teacher_comparison.png)

#### Knowledge Distillation Effectiveness
![KD Effectiveness](artifacts/imgs/01_baselines/kd_effectiveness.png)

#### Latency Benchmarks
![Latency Benchmarks](artifacts/imgs/01_baselines/latency_benchmarks.png)

</details>

<details>
<summary><b>📈 ROC & PR Curves (click to expand)</b></summary>

#### Holdout Set Evaluation
![Holdout Evaluation](artifacts/imgs/01_baselines/holdout_evaluation.png)

#### Teacher Threshold Curves
![Teacher Threshold Curves](artifacts/imgs/01_baselines/teacher_threshold_curves.png)

#### Student Threshold Curves
![Student Threshold Curves](artifacts/imgs/01_baselines/student_threshold_curves.png)

</details>

<details>
<summary><b>🎯 Best Student Model Training (click to expand)</b></summary>

#### Training Curves (Student T=1, α=0.5)
![Training Curves](artifacts/imgs/training/student_T1.0_alpha0.5/training_curves.png)

#### ROC & PR Curves
![ROC PR Curves](artifacts/imgs/training/student_T1.0_alpha0.5/roc_pr_curves.png)

#### Reliability Diagram (Calibration)
![Reliability Diagram](artifacts/imgs/training/student_T1.0_alpha0.5/reliability_diagram.png)

#### Teacher vs Student Comparison
![Model Comparison](artifacts/imgs/training/student_T1.0_alpha0.5/model_comparison.png)

</details>

<details>
<summary><b>🔬 Inference Analysis (click to expand)</b></summary>

#### Teacher vs Student Predictions
![Teacher vs Student](artifacts/imgs/02_inference/teacher_vs_student.png)

#### Confidence Distribution
![Confidence Distribution](artifacts/imgs/02_inference/confidence_distribution.png)

#### Challenging Cases
![Challenging Cases](artifacts/imgs/02_inference/challenging_cases.png)

#### High Confidence Errors
![High Confidence Errors](artifacts/imgs/02_inference/high_confidence_errors.png)

</details>

### Sklearn Baselines

Traditional ML baselines for comparison (using hand-crafted features):

| Model | Features | ROC-AUC | PR-AUC |
|-------|----------|---------|--------|
| Random Forest | Combined | 0.853 | 0.392 |
| Gradient Boosting | Combined | 0.845 | 0.366 |
| SVM (RBF) | Combined | 0.824 | 0.336 |
| Logistic Regression | Combined | 0.797 | 0.289 |

Run baselines with:

```bash
make sklearn-baselines        # Full benchmark (all models, combined features)
make sklearn-baselines-quick  # Quick test (logistic regression, 1000 samples)
```

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
make summary       # Generate experiment status report
```

## Changelog

- **Lesion-aware data splitting**: Prevents data leakage by ensuring images from the same lesion don't appear in different splits
- **Teacher models**: Multiple architectures supported for ablation:
  - **ResNet family**: ResNet-18, 34, 50, 101, 152
  - **EfficientNet family**: EfficientNet-B0 through B7
  - All pretrained on ImageNet, fine-tuned with focal loss for class imbalance
- **Student model**: MobileNetV3-Small for mobile deployment (<25 MB)
- **Knowledge distillation**: Temperature-scaled KD with focused hyperparameter search (T ∈ {1, 2}, α ∈ {0.5, 0.9})
- **Comprehensive evaluation**: ROC-AUC, PR-AUC, sensitivity/specificity at 95% recall, ECE calibration
- **INT8 quantization**: Post-training quantization for edge deployment
- **Sklearn baselines**: Traditional ML benchmarks (Logistic Regression, Random Forest, GBM, SVM)
- **Experiment tracking**: W&B integration for logging, `make summary` for experiment status
