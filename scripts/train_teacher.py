"""Train teacher model for melanoma detection.

Usage:
    python scripts/train_teacher.py --architecture resnet34 --epochs 50
    python scripts/train_teacher.py --config artifacts/configs/teacher.yaml
"""

import argparse
import logging
import pathlib
import sys

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import torch

from src.config import (
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    DataConfig,
    TeacherConfig,
    TrainingConfig,
    WandbConfig,
    get_device,
    set_seed,
)
from src.data.dataset import (
    HAM10000Dataset,
    create_dataloaders,
    get_eval_transforms,
)
from src.data.splits import load_or_create_splits
from src.evaluation.metrics import compute_deployment_metrics, evaluate_model
from src.models.architectures import TeacherModel
from src.plotting.training_plots import (
    plot_reliability_diagram,
    plot_roc_pr_curves,
    plot_training_curves,
)
from src.training.trainer import TeacherTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "train_teacher.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train teacher model")

    # Supported architectures
    resnet_archs = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    efficientnet_archs = [f"efficientnet_b{i}" for i in range(8)]  # B0-B7
    all_archs = resnet_archs + efficientnet_archs

    # Model
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet34",
        choices=all_archs,
        help="Teacher architecture",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--loss",
        type=str,
        default="focal",
        choices=["bce", "weighted_bce", "focal"],
        help="Loss function",
    )
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Data
    parser.add_argument("--recreate-splits", action="store_true", help="Recreate data splits")
    parser.add_argument("--weighted-sampling", action="store_true", help="Use weighted sampling")
    parser.add_argument(
        "--augmentation",
        type=str,
        default="standard",
        choices=["light", "standard", "heavy", "dermoscopy"],
        help="Augmentation level: light, standard, heavy, or dermoscopy (domain-specific)",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Experiment name
    exp_name = args.name or f"teacher_{args.architecture}_{args.loss}"
    logger.info(f"Experiment: {exp_name}")

    # Create configurations
    data_config = DataConfig(batch_size=args.batch_size)
    teacher_config = TeacherConfig(
        architecture=args.architecture,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        early_stopping_patience=args.patience,
    )

    # Load or create data splits
    logger.info("Loading data splits...")
    train_path, val_path, holdout_path = load_or_create_splits(
        force_recreate=args.recreate_splits,
        random_seed=args.seed,
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path,
        val_path,
        config=data_config,
        use_weighted_sampling=args.weighted_sampling,
        augmentation_level=args.augmentation,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Augmentation level: {args.augmentation}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Get positive weight for loss
    train_dataset = train_loader.dataset
    pos_weight = (
        train_dataset.get_pos_weight() if hasattr(train_dataset, "get_pos_weight") else None
    )
    if pos_weight is not None:
        logger.info(f"Positive class weight: {pos_weight.item():.2f}")

    # Create model
    logger.info(f"Creating {args.architecture} teacher model...")
    model = TeacherModel(teacher_config)

    params = model.count_parameters()
    logger.info(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # W&B config
    wandb_config = WandbConfig(enabled=not args.no_wandb) if not args.no_wandb else None

    # Create trainer
    trainer = TeacherTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        experiment_name=exp_name,
        pos_weight=pos_weight,
        wandb_config=wandb_config,
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    logger.info(f"Best epoch: {history.best_epoch}")
    logger.info(f"Best ROC-AUC: {history.best_val_roc_auc:.4f}")
    logger.info(f"Best F1: {history.best_val_f1:.4f}")

    # Plot training curves
    fig_dir = FIGURES_DIR / "training" / exp_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(
        history.to_dict(),
        title=f"Teacher Training: {args.architecture}",
        save_path=fig_dir / "training_curves.png",
    )

    # Load best model and evaluate on holdout
    logger.info("Evaluating best model on holdout set...")
    best_ckpt = CHECKPOINTS_DIR / f"{exp_name}_best.pth"
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create holdout loader
    holdout_dataset = HAM10000Dataset(
        holdout_path,
        transform=get_eval_transforms(data_config),
        config=data_config,
    )
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )

    # Evaluate
    holdout_metrics = evaluate_model(model, holdout_loader, device)

    logger.info("=" * 60)
    logger.info("HOLDOUT SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"ROC-AUC: {holdout_metrics.roc_auc:.4f}")
    logger.info(f"PR-AUC: {holdout_metrics.pr_auc:.4f}")
    logger.info(f"F1: {holdout_metrics.f1:.4f}")
    logger.info(f"Sensitivity: {holdout_metrics.recall:.4f}")
    logger.info(f"Specificity: {holdout_metrics.specificity:.4f}")
    logger.info(f"ECE: {holdout_metrics.ece:.4f}")
    logger.info(f"Specificity @95% sens: {holdout_metrics.specificity_at_target_sens:.4f}")
    logger.info(f"PPV @95% sens: {holdout_metrics.ppv_at_target_sens:.4f}")
    logger.info(f"NPV @95% sens: {holdout_metrics.npv_at_target_sens:.4f}")

    # Save metrics
    holdout_metrics.to_json(fig_dir / "holdout_metrics.json")

    # Generate reliability diagram
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in holdout_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    plot_reliability_diagram(
        y_true,
        y_prob,
        title=f"Teacher Reliability Diagram: {args.architecture}",
        save_path=fig_dir / "reliability_diagram.png",
    )

    plot_roc_pr_curves(
        y_true,
        y_prob,
        title=f"Teacher ROC/PR Curves: {args.architecture}",
        save_path=fig_dir / "roc_pr_curves.png",
    )

    # Deployment metrics
    deployment = compute_deployment_metrics(model, device=device)
    logger.info(f"Model size: {deployment.model_size_mb:.2f} MB")
    logger.info(f"Inference latency: {deployment.avg_latency_ms:.2f} ms")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
