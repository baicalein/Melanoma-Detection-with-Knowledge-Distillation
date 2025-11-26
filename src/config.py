"""Centralized configuration for the HAM10000 melanoma detection project.

This module defines all hyperparameters, paths, and experimental settings
in a single location for reproducibility and ease of modification.
"""

import os
import pathlib
import random
from dataclasses import dataclass, field

import numpy as np
import torch

# ============================================================================
# PATHS
# ============================================================================
ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ham_1000_archive"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = MODELS_DIR / "logs"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "imgs"
TABLES_DIR = ARTIFACTS_DIR / "tbls"

# Ensure directories exist
for d in [CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# DATA CONFIGURATION
# ============================================================================
@dataclass
class DataConfig:
    """Data loading and splitting configuration."""

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    holdout_ratio: float = 0.15

    # Image preprocessing
    image_size: int = 224
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation settings
    use_augmentation: bool = True
    random_crop_scale: tuple[float, float] = (0.8, 1.0)
    rotation_degrees: int = 30
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.15
    color_jitter_hue: float = 0.05
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    random_erasing_prob: float = 0.2


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
@dataclass
class TeacherConfig:
    """Teacher model configuration."""

    # Architecture choices: resnet18, resnet34, resnet50
    architecture: str = "resnet34"
    pretrained: bool = True
    num_classes: int = 1  # Binary: single logit output
    dropout: float = 0.3

    # Fine-tuning strategy
    freeze_backbone: bool = False  # Full fine-tuning recommended
    unfreeze_layers: list[str] = field(default_factory=lambda: ["layer3", "layer4", "fc"])


@dataclass
class StudentConfig:
    """Student model configuration for mobile deployment."""

    # Architecture: MobileNetV3-Small for mobile deployment
    architecture: str = "mobilenet_v3_small"
    pretrained: bool = True
    num_classes: int = 1
    dropout: float = 0.2

    # Target deployment constraints
    max_size_mb: float = 25.0  # Mobile target
    max_edge_size_mb: float = 2.0  # Edge device target


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Epochs
    max_epochs: int = 50
    early_stopping_patience: int = 10

    # Optimizer (AdamW recommended)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Learning rate scheduler
    scheduler: str = "cosine"  # Options: "cosine", "plateau", "step"
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # Loss function
    loss_type: str = "focal"  # Options: "bce", "weighted_bce", "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75  # Weight for positive class (melanoma)

    # Gradient clipping
    gradient_clip_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True


# ============================================================================
# KNOWLEDGE DISTILLATION CONFIGURATION
# ============================================================================
@dataclass
class KDConfig:
    """Knowledge distillation hyperparameters.

    Based on feedback: Focus on small, well-motivated search space.
    T ∈ {1, 2}, ω ∈ {0.5, 0.9}
    """

    # Temperature for softening logits
    temperature: float = 2.0  # T ∈ {1, 2}

    # Weight for KD loss vs hard label loss
    # L = ω * L_KD + (1 - ω) * L_BCE
    alpha: float = 0.5  # ω ∈ {0.5, 0.9}

    # KD loss type
    loss_type: str = "kl_div"  # Options: "kl_div", "mse"


# ============================================================================
# QUANTIZATION CONFIGURATION
# ============================================================================
@dataclass
class QuantizationConfig:
    """Post-training quantization settings."""

    # Quantization backend
    backend: str = "qnnpack"  # For mobile: "qnnpack", for server: "fbgemm"

    # Quantization type
    dtype: str = "qint8"

    # Quantization-aware training (only if PTQ degrades AUC > 0.02)
    use_qat: bool = False
    qat_epochs: int = 5


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
@dataclass
class EvalConfig:
    """Evaluation and metrics configuration."""

    # Operating point for clinical metrics
    target_sensitivity: float = 0.95

    # Calibration
    ece_bins: int = 15

    # Inference benchmarking
    warmup_iterations: int = 10
    benchmark_iterations: int = 100


# ============================================================================
# WANDB CONFIGURATION
# ============================================================================
WANDB_KEY_PATH = ROOT / "keys" / "wandb_key.txt"


def load_wandb_key() -> str | None:
    """Load W&B API key from file."""
    if WANDB_KEY_PATH.exists():
        return WANDB_KEY_PATH.read_text().strip()
    return os.getenv("WANDB_API_KEY")


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    project: str = "melanoma_kd_ds6050"
    entity: str | None = None
    enabled: bool = True
    log_model: bool = True
    log_freq: int = 10

    def init_wandb(self, run_name: str, config: dict = None, tags: list[str] = None):
        """Initialize W&B run with API key from file."""
        if not self.enabled:
            return None

        try:
            import wandb

            api_key = load_wandb_key()
            if api_key:
                wandb.login(key=api_key)

            run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config or {},
                tags=tags or [],
                reinit=True,
            )
            return run
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            return None
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            return None


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str = "baseline"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    kd: KDConfig = field(default_factory=KDConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    seed: int = RANDOM_SEED
    device: str = field(default_factory=lambda: get_device())


def get_device() -> str:
    """Detect best available device. Prefer MPS (Apple Silicon) over CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================


def get_teacher_training_config() -> ExperimentConfig:
    """Configuration for training the teacher model."""
    config = ExperimentConfig(
        name="teacher_resnet34",
        description="ResNet34 teacher with focal loss for melanoma detection",
        tags=["teacher", "resnet34", "focal_loss"],
    )
    config.training.max_epochs = 50
    config.training.loss_type = "focal"
    return config


def get_kd_config(temperature: float = 2.0, alpha: float = 0.5) -> ExperimentConfig:
    """Configuration for knowledge distillation experiments."""
    config = ExperimentConfig(
        name=f"kd_T{temperature}_alpha{alpha}",
        description=f"KD with T={temperature}, α={alpha}",
        tags=["kd", "mobilenetv3", f"T{temperature}", f"alpha{alpha}"],
    )
    config.kd.temperature = temperature
    config.kd.alpha = alpha
    return config


def get_quantization_config() -> ExperimentConfig:
    """Configuration for quantization experiments."""
    config = ExperimentConfig(
        name="quantization_int8",
        description="Post-training INT8 quantization",
        tags=["quantization", "int8", "mobile"],
    )
    return config
