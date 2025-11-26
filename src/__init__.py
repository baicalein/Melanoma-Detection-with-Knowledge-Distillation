"""
HAM10000 Melanoma Detection with Knowledge Distillation.

This package provides:
- Data loading and lesion-aware splitting for HAM10000
- Teacher (ResNet) and Student (MobileNetV3) model architectures
- Knowledge distillation with focal loss
- Comprehensive evaluation metrics (ROC-AUC, PR-AUC, ECE, calibration)
- Post-training quantization for mobile deployment

Usage:
    from src.config import ExperimentConfig, set_seed
    from src.data import HAM10000Dataset, load_or_create_splits
    from src.models import TeacherModel, StudentModel
    from src.training import TeacherTrainer, StudentTrainer
    from src.evaluation import evaluate_model, compute_deployment_metrics
"""

from src.config import (
    ExperimentConfig,
    DataConfig,
    TeacherConfig,
    StudentConfig,
    TrainingConfig,
    KDConfig,
    WandbConfig,
    set_seed,
    get_device,
    RANDOM_SEED,
)

__version__ = "0.2.0"

__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "TeacherConfig",
    "StudentConfig",
    "TrainingConfig",
    "KDConfig",
    "WandbConfig",
    "set_seed",
    "get_device",
    "RANDOM_SEED",
]
