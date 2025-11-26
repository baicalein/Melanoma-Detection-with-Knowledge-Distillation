"""
Train student model with knowledge distillation.

Usage:
    python scripts/train_student.py --teacher-ckpt models/checkpoints/teacher_resnet34_best.pth
    python scripts/train_student.py --temperature 2 --alpha 0.5
"""

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import torch

from src.config import (
    StudentConfig,
    TeacherConfig,
    TrainingConfig,
    KDConfig,
    DataConfig,
    WandbConfig,
    PROCESSED_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    FIGURES_DIR,
    set_seed,
    get_device,
)
from src.data.dataset import HAM10000Dataset, get_train_transforms, get_eval_transforms, create_dataloaders
from src.data.splits import load_or_create_splits
from src.models.architectures import TeacherModel, StudentModel, load_teacher_checkpoint
from src.training.trainer import StudentTrainer
from src.evaluation.metrics import evaluate_model, compute_deployment_metrics
from src.plotting.training_plots import (
    plot_training_curves,
    plot_reliability_diagram,
    plot_roc_pr_curves,
    plot_model_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "train_student.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train student with KD")
    
    # Teacher
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        required=True,
        help="Path to teacher checkpoint",
    )
    parser.add_argument(
        "--teacher-arch",
        type=str,
        default="resnet34",
        help="Teacher architecture",
    )
    
    # Student
    parser.add_argument(
        "--student-arch",
        type=str,
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "mobilenet_v3_large"],
        help="Student architecture",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Student dropout")
    
    # KD hyperparameters (focused search space per feedback)
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        choices=[1.0, 2.0],
        help="KD temperature (T ∈ {1, 2})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="KD alpha weight (ω ∈ {0.5, 0.9})",
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Experiment name
    exp_name = args.name or f"student_T{args.temperature}_alpha{args.alpha}"
    logger.info(f"Experiment: {exp_name}")
    
    # Configurations
    data_config = DataConfig(batch_size=args.batch_size)
    teacher_config = TeacherConfig(architecture=args.teacher_arch)
    student_config = StudentConfig(
        architecture=args.student_arch,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
    )
    kd_config = KDConfig(
        temperature=args.temperature,
        alpha=args.alpha,
    )
    
    # Load data
    train_path, val_path, holdout_path = load_or_create_splits()
    train_loader, val_loader = create_dataloaders(
        train_path, val_path, config=data_config
    )
    
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Load teacher
    logger.info(f"Loading teacher from {args.teacher_ckpt}...")
    teacher, _ = load_teacher_checkpoint(
        args.teacher_ckpt, config=teacher_config, device=device
    )
    teacher.eval()
    teacher_params = teacher.count_parameters()
    logger.info(f"Teacher params: {teacher_params['total']:,}")
    
    # Create student
    logger.info(f"Creating {args.student_arch} student...")
    student = StudentModel(student_config)
    student_params = student.count_parameters()
    logger.info(f"Student params: {student_params['total']:,}")
    logger.info(f"Compression ratio: {teacher_params['total'] / student_params['total']:.1f}x")
    
    # Check deployment constraints
    constraints = student.check_deployment_constraints()
    logger.info(f"Student size: {constraints['size_mb']:.2f} MB")
    logger.info(f"Meets mobile constraint (<{constraints['mobile_target_mb']} MB): {constraints['meets_mobile_constraint']}")
    
    # W&B config
    wandb_config = WandbConfig(enabled=not args.no_wandb) if not args.no_wandb else None
    
    # Create trainer
    trainer = StudentTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        kd_config=kd_config,
        device=device,
        experiment_name=exp_name,
        wandb_config=wandb_config,
    )
    
    # Train
    logger.info("Starting KD training...")
    history = trainer.train()
    
    logger.info(f"Best epoch: {history.best_epoch}")
    logger.info(f"Best ROC-AUC: {history.best_val_roc_auc:.4f}")
    
    # Plot training curves
    fig_dir = FIGURES_DIR / "training" / exp_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_curves(
        history.to_dict(),
        title=f"KD Training: T={args.temperature}, α={args.alpha}",
        save_path=fig_dir / "training_curves.png",
    )
    
    # Load best student and evaluate
    logger.info("Evaluating best student on holdout...")
    best_ckpt = CHECKPOINTS_DIR / f"{exp_name}_best.pth"
    checkpoint = torch.load(best_ckpt, map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])
    
    # Holdout loader
    holdout_dataset = HAM10000Dataset(
        holdout_path,
        transform=get_eval_transforms(data_config),
    )
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
    )
    
    # Evaluate both teacher and student on holdout
    def get_predictions(model, loader):
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(targets.numpy())
        return np.concatenate(all_targets), np.concatenate(all_probs)
    
    teacher_true, teacher_prob = get_predictions(teacher, holdout_loader)
    student_true, student_prob = get_predictions(student, holdout_loader)
    
    teacher_metrics = evaluate_model(teacher, holdout_loader, device)
    student_metrics = evaluate_model(student, holdout_loader, device)
    
    # Log results
    logger.info("=" * 60)
    logger.info("HOLDOUT RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<25} {'Teacher':>12} {'Student':>12} {'Δ':>10}")
    logger.info("-" * 60)
    logger.info(f"{'ROC-AUC':<25} {teacher_metrics.roc_auc:>12.4f} {student_metrics.roc_auc:>12.4f} {student_metrics.roc_auc - teacher_metrics.roc_auc:>+10.4f}")
    logger.info(f"{'PR-AUC':<25} {teacher_metrics.pr_auc:>12.4f} {student_metrics.pr_auc:>12.4f} {student_metrics.pr_auc - teacher_metrics.pr_auc:>+10.4f}")
    logger.info(f"{'F1':<25} {teacher_metrics.f1:>12.4f} {student_metrics.f1:>12.4f} {student_metrics.f1 - teacher_metrics.f1:>+10.4f}")
    logger.info(f"{'ECE':<25} {teacher_metrics.ece:>12.4f} {student_metrics.ece:>12.4f} {student_metrics.ece - teacher_metrics.ece:>+10.4f}")
    logger.info(f"{'Spec @95% sens':<25} {teacher_metrics.specificity_at_target_sens:>12.4f} {student_metrics.specificity_at_target_sens:>12.4f} {student_metrics.specificity_at_target_sens - teacher_metrics.specificity_at_target_sens:>+10.4f}")
    
    # Save metrics
    student_metrics.to_json(fig_dir / "student_holdout_metrics.json")
    teacher_metrics.to_json(fig_dir / "teacher_holdout_metrics.json")
    
    # Reliability diagrams
    plot_reliability_diagram(
        student_true, student_prob,
        title=f"Student Reliability: T={args.temperature}, α={args.alpha}",
        save_path=fig_dir / "reliability_diagram.png",
    )
    
    plot_roc_pr_curves(
        student_true, student_prob,
        title=f"Student ROC/PR: T={args.temperature}, α={args.alpha}",
        save_path=fig_dir / "roc_pr_curves.png",
    )
    
    # Model comparison
    plot_model_comparison(
        {
            "Teacher": teacher_metrics.to_dict(),
            "Student": student_metrics.to_dict(),
        },
        save_path=fig_dir / "model_comparison.png",
    )
    
    # Deployment metrics
    teacher_deployment = compute_deployment_metrics(teacher, device=device)
    student_deployment = compute_deployment_metrics(student, device=device)
    
    logger.info("=" * 60)
    logger.info("DEPLOYMENT METRICS")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<25} {'Teacher':>12} {'Student':>12}")
    logger.info("-" * 60)
    logger.info(f"{'Size (MB)':<25} {teacher_deployment.model_size_mb:>12.2f} {student_deployment.model_size_mb:>12.2f}")
    logger.info(f"{'Latency (ms)':<25} {teacher_deployment.avg_latency_ms:>12.2f} {student_deployment.avg_latency_ms:>12.2f}")
    logger.info(f"{'Throughput (img/s)':<25} {teacher_deployment.throughput_images_per_sec:>12.1f} {student_deployment.throughput_images_per_sec:>12.1f}")
    
    logger.info("KD training complete!")


if __name__ == "__main__":
    main()
