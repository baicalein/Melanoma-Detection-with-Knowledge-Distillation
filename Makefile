# Makefile for HAM10000 Melanoma Detection with Knowledge Distillation
# Provides convenient commands for environment setup, training, and evaluation

.PHONY: help install install-dev test lint format clean \
        data splits train-teacher train-student train-all \
        quantize evaluate run-ablation check-all

# ============================================================================
# Configuration
# ============================================================================
PYTHON := uv run python
SEED := 42
EPOCHS := 50
BATCH_SIZE := 32
TEACHER_ARCH := resnet34
STUDENT_ARCH := mobilenet_v3_small
TEACHER_CKPT := models/checkpoints/teacher_$(TEACHER_ARCH)_focal_best.pth
STUDENT_CKPT := models/checkpoints/student_T2_alpha0.5_best.pth

# KD Hyperparameters (focused search space)
KD_TEMP := 2.0
KD_ALPHA := 0.5

# ============================================================================
# Help
# ============================================================================
help:
	@echo "============================================================================"
	@echo "HAM10000 Melanoma Detection - Knowledge Distillation Pipeline"
	@echo "============================================================================"
	@echo ""
	@echo "SETUP:"
	@echo "  make install          - Install production dependencies with uv"
	@echo "  make install-dev      - Install with dev dependencies"
	@echo "  make env              - Create/activate venv and install deps"
	@echo ""
	@echo "DATA:"
	@echo "  make data             - Download and prepare HAM10000 dataset"
	@echo "  make splits           - Create lesion-aware train/val/holdout splits"
	@echo ""
	@echo "TRAINING:"
	@echo "  make train-teacher    - Train teacher model (ResNet34 + focal loss)"
	@echo "  make train-student    - Train student with KD (MobileNetV3)"
	@echo "  make train-all        - Run full training pipeline"
	@echo "  make run-ablation     - Run KD ablation (T∈{1,2}, α∈{0.5,0.9})"
	@echo ""
	@echo "EVALUATION:"
	@echo "  make quantize         - Quantize student model to INT8"
	@echo "  make evaluate         - Evaluate all models on holdout set"
	@echo ""
	@echo "DEVELOPMENT:"
	@echo "  make test             - Run unit tests"
	@echo "  make lint             - Run linters (ruff)"
	@echo "  make format           - Format code"
	@echo "  make clean            - Clean build artifacts"
	@echo "  make check-all        - Run format, lint, and test"
	@echo ""
	@echo "NOTEBOOKS:"
	@echo "  make notebook         - Start Jupyter Lab"
	@echo "  make eda              - Run EDA notebook"
	@echo ""

# ============================================================================
# Environment Setup
# ============================================================================
install:
	uv sync
	@echo "✓ Dependencies installed"

install-dev:
	uv sync --extra dev
	@echo "✓ Dev dependencies installed"

env:
	@echo "Creating virtual environment..."
	uv venv
	@echo "Installing dependencies..."
	uv sync --extra dev
	@echo ""
	@echo "✓ Environment ready!"
	@echo "  Activate with: source .venv/bin/activate"
	@echo "  Or use 'uv run' to run commands in the environment"

# ============================================================================
# Data Preparation
# ============================================================================
data:
	@echo "Preparing HAM10000 dataset..."
	@if [ ! -d "data/raw/ham_1000_archive/images" ]; then \
		echo "ERROR: HAM10000 images not found."; \
		echo "Please download from Kaggle and extract to data/raw/ham_1000_archive/"; \
		exit 1; \
	fi
	$(PYTHON) -c "from src.data.build_data import create_base_df; create_base_df()"
	@echo "✓ Dataset prepared: data/processed/labeled_ham10000.csv"

splits: data
	@echo "Creating lesion-aware data splits (70/15/15)..."
	$(PYTHON) -c "from src.data.splits import load_or_create_splits; load_or_create_splits(force_recreate=True)"
	@echo "✓ Splits created:"
	@echo "  - data/processed/train_data.csv"
	@echo "  - data/processed/val_data.csv"
	@echo "  - data/processed/holdout_data.csv"

verify-splits:
	@echo "Verifying no lesion leakage across splits..."
	$(PYTHON) -c "from src.data.splits import verify_no_leakage; from src.config import PROCESSED_DIR; \
		verify_no_leakage(PROCESSED_DIR/'train_data.csv', PROCESSED_DIR/'val_data.csv', PROCESSED_DIR/'holdout_data.csv')"

# ============================================================================
# Training
# ============================================================================
train-teacher: splits
	@echo "============================================================================"
	@echo "Training Teacher Model: $(TEACHER_ARCH)"
	@echo "============================================================================"
	$(PYTHON) scripts/train_teacher.py \
		--architecture $(TEACHER_ARCH) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr 1e-4 \
		--loss focal \
		--patience 10 \
		--seed $(SEED)
	@echo "✓ Teacher training complete: $(TEACHER_CKPT)"

train-student: train-teacher
	@echo "============================================================================"
	@echo "Training Student Model with KD: T=$(KD_TEMP), α=$(KD_ALPHA)"
	@echo "============================================================================"
	$(PYTHON) scripts/train_student.py \
		--teacher-ckpt $(TEACHER_CKPT) \
		--teacher-arch $(TEACHER_ARCH) \
		--student-arch $(STUDENT_ARCH) \
		--temperature $(KD_TEMP) \
		--alpha $(KD_ALPHA) \
		--epochs $(EPOCHS) \
		--seed $(SEED)
	@echo "✓ Student training complete"

train-all: train-teacher train-student quantize
	@echo "============================================================================"
	@echo "✓ Full training pipeline complete!"
	@echo "============================================================================"
	@echo "Outputs:"
	@echo "  - Teacher: $(TEACHER_CKPT)"
	@echo "  - Student: $(STUDENT_CKPT)"
	@echo "  - Quantized: models/checkpoints/quantized_dynamic_quantized.pth"
	@echo "  - Figures: artifacts/imgs/training/"

run-ablation: train-teacher
	@echo "============================================================================"
	@echo "Running KD Ablation: T∈{1,2}, α∈{0.5,0.9}"
	@echo "============================================================================"
	$(PYTHON) scripts/run_kd_ablation.py \
		--teacher-ckpt $(TEACHER_CKPT) \
		--epochs $(EPOCHS) \
		--seed $(SEED)
	@echo "✓ Ablation complete. Results in artifacts/imgs/kd_ablation_summary.json"

# ============================================================================
# Quantization & Evaluation
# ============================================================================
quantize:
	@echo "Quantizing student model to INT8..."
	$(PYTHON) scripts/quantize_model.py \
		--model-ckpt $(STUDENT_CKPT) \
		--student-arch $(STUDENT_ARCH) \
		--method dynamic
	@echo "✓ Quantization complete"

evaluate:
	@echo "Evaluating all models on holdout set..."
	$(PYTHON) -c "\
from src.config import PROCESSED_DIR, CHECKPOINTS_DIR, get_device; \
from src.data.dataset import HAM10000Dataset, get_eval_transforms; \
from src.models.architectures import TeacherModel, StudentModel, load_teacher_checkpoint, load_student_checkpoint; \
from src.evaluation.metrics import evaluate_model, compute_deployment_metrics; \
import torch; \
device = get_device(); \
print(f'Device: {device}'); \
holdout = torch.utils.data.DataLoader(HAM10000Dataset(PROCESSED_DIR/'holdout_data.csv', transform=get_eval_transforms()), batch_size=32); \
print('\\n=== Teacher ==='); \
teacher, _ = load_teacher_checkpoint(str(CHECKPOINTS_DIR/'teacher_resnet34_focal_best.pth'), device=device); \
m = evaluate_model(teacher, holdout, device); \
print(f'ROC-AUC: {m.roc_auc:.4f}, ECE: {m.ece:.4f}'); \
print('\\n=== Student ==='); \
student, _ = load_student_checkpoint(str(CHECKPOINTS_DIR/'student_T2_alpha0.5_best.pth'), device=device); \
m = evaluate_model(student, holdout, device); \
print(f'ROC-AUC: {m.roc_auc:.4f}, ECE: {m.ece:.4f}'); \
"

# ============================================================================
# Development
# ============================================================================
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-quick:
	$(PYTHON) -m pytest tests/test_core.py -v -x

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "✓ Cleaned build artifacts"

check-all: format lint test
	@echo "✓ All checks passed!"

# ============================================================================
# Notebooks
# ============================================================================
notebook:
	uv run jupyter lab

eda:
	uv run jupyter nbconvert --execute notebooks/00_eda.ipynb --to html
	@echo "✓ EDA notebook executed: notebooks/00_eda.html"

# ============================================================================
# Utilities
# ============================================================================
tree:
	tree -I '__pycache__|*.pyc|.git|.venv|*.egg-info|.pytest_cache|.ruff_cache|.ipynb_checkpoints' -L 3

gpu-check:
	$(PYTHON) -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}'); print(f'Device: {\"cuda\" if torch.cuda.is_available() else (\"mps\" if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available() else \"cpu\")}')"

show-config:
	$(PYTHON) -c "from src.config import ExperimentConfig; import json; c = ExperimentConfig(); print(json.dumps({k: str(v) for k, v in c.__dict__.items()}, indent=2))"

# ============================================================================
# Quick Start (run everything)
# ============================================================================
all: env splits train-all evaluate
	@echo ""
	@echo "============================================================================"
	@echo "✓ COMPLETE PIPELINE FINISHED"
	@echo "============================================================================"
	@echo ""
	@echo "Results:"
	@echo "  - Checkpoints: models/checkpoints/"
	@echo "  - Figures: artifacts/imgs/"
	@echo "  - Logs: models/logs/"
	@echo ""
