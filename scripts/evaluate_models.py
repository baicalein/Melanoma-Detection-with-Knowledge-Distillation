"""Evaluate all available model checkpoints on the holdout set.

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --output artifacts/tbls/evaluation_results.csv
"""

import argparse
import json
import logging
import pathlib
import sys

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import torch

from src.config import (
    CHECKPOINTS_DIR,
    PROCESSED_DIR,
    TABLES_DIR,
    DataConfig,
    StudentConfig,
    TeacherConfig,
    get_device,
)
from src.data.dataset import HAM10000Dataset, get_eval_transforms
from src.evaluation.metrics import compute_deployment_metrics, evaluate_model
from src.models.architectures import StudentModel, TeacherModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# All supported teacher architectures
RESNET_ARCHS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
EFFICIENTNET_ARCHS = [f"efficientnet_b{i}" for i in range(8)]
ALL_TEACHER_ARCHS = RESNET_ARCHS + EFFICIENTNET_ARCHS

# Student KD configurations
KD_CONFIGS = [
    {"temperature": 1, "alpha": 0.5},
    {"temperature": 1, "alpha": 0.9},
    {"temperature": 2, "alpha": 0.5},
    {"temperature": 2, "alpha": 0.9},
]


def find_teacher_checkpoint(arch: str) -> pathlib.Path | None:
    """Find checkpoint for a teacher architecture."""
    patterns = [
        CHECKPOINTS_DIR / f"{arch}_best.pth",
        CHECKPOINTS_DIR / f"teacher_{arch}_focal_best.pth",
        CHECKPOINTS_DIR / f"teacher_{arch}_best.pth",
    ]
    for p in patterns:
        if p.exists():
            return p
    return None


def find_student_checkpoint(temperature: int, alpha: float) -> pathlib.Path | None:
    """Find checkpoint for a student KD configuration."""
    patterns = [
        CHECKPOINTS_DIR / f"student_T{temperature}_alpha{alpha}_best.pth",
        CHECKPOINTS_DIR / f"student_T{temperature}_alpha{alpha:.1f}_best.pth",
    ]
    for p in patterns:
        if p.exists():
            return p
    return None


def evaluate_teacher(
    arch: str,
    checkpoint_path: pathlib.Path,
    holdout_loader: torch.utils.data.DataLoader,
    device: str,
) -> dict:
    """Evaluate a teacher model."""
    config = TeacherConfig(architecture=arch)
    model = TeacherModel(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    metrics = evaluate_model(model, holdout_loader, device)
    deployment = compute_deployment_metrics(model, device=device)

    return {
        "model_type": "teacher",
        "architecture": arch,
        "checkpoint": str(checkpoint_path.name),
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "f1": metrics.f1,
        "sensitivity": metrics.recall,
        "specificity": metrics.specificity,
        "ece": metrics.ece,
        "specificity_at_95_sens": metrics.specificity_at_target_sens,
        "ppv_at_95_sens": metrics.ppv_at_target_sens,
        "npv_at_95_sens": metrics.npv_at_target_sens,
        "size_mb": deployment.model_size_mb,
        "latency_ms": deployment.avg_latency_ms,
    }


def evaluate_student(
    temperature: int,
    alpha: float,
    checkpoint_path: pathlib.Path,
    holdout_loader: torch.utils.data.DataLoader,
    device: str,
) -> dict:
    """Evaluate a student model."""
    config = StudentConfig()
    model = StudentModel(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    metrics = evaluate_model(model, holdout_loader, device)
    deployment = compute_deployment_metrics(model, device=device)

    return {
        "model_type": "student",
        "architecture": f"mobilenet_v3_small (T={temperature}, α={alpha})",
        "checkpoint": str(checkpoint_path.name),
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "f1": metrics.f1,
        "sensitivity": metrics.recall,
        "specificity": metrics.specificity,
        "ece": metrics.ece,
        "specificity_at_95_sens": metrics.specificity_at_target_sens,
        "ppv_at_95_sens": metrics.ppv_at_target_sens,
        "npv_at_95_sens": metrics.npv_at_target_sens,
        "size_mb": deployment.model_size_mb,
        "latency_ms": deployment.avg_latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all available models")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for results",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Output JSON path for results",
    )
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Create holdout dataloader
    data_config = DataConfig()
    holdout_path = PROCESSED_DIR / "holdout_data.csv"

    if not holdout_path.exists():
        logger.error(f"Holdout data not found: {holdout_path}")
        logger.error("Run 'make splits' first to create data splits.")
        sys.exit(1)

    holdout_dataset = HAM10000Dataset(
        holdout_path,
        transform=get_eval_transforms(data_config),
        config=data_config,
    )
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
    )

    logger.info(f"Holdout samples: {len(holdout_dataset)}")

    results = []

    # Evaluate teachers
    print("\n" + "=" * 70)
    print("TEACHER MODELS")
    print("=" * 70)

    for arch in ALL_TEACHER_ARCHS:
        ckpt = find_teacher_checkpoint(arch)
        if ckpt:
            print(f"\n>>> Evaluating {arch}...")
            try:
                result = evaluate_teacher(arch, ckpt, holdout_loader, device)
                results.append(result)
                print(f"    ROC-AUC: {result['roc_auc']:.4f}, ECE: {result['ece']:.4f}, Size: {result['size_mb']:.1f} MB")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        else:
            print(f"\n⏭  Skipping {arch} - no checkpoint found")

    # Evaluate students
    print("\n" + "=" * 70)
    print("STUDENT MODELS (Knowledge Distillation)")
    print("=" * 70)

    for kd_cfg in KD_CONFIGS:
        t, a = kd_cfg["temperature"], kd_cfg["alpha"]
        ckpt = find_student_checkpoint(t, a)
        if ckpt:
            print(f"\n>>> Evaluating student T={t}, α={a}...")
            try:
                result = evaluate_student(t, a, ckpt, holdout_loader, device)
                results.append(result)
                print(f"    ROC-AUC: {result['roc_auc']:.4f}, ECE: {result['ece']:.4f}, Size: {result['size_mb']:.1f} MB")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        else:
            print(f"\n⏭  Skipping student T={t}, α={a} - no checkpoint found")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        df = pd.DataFrame(results)

        # Sort by ROC-AUC descending
        df = df.sort_values("roc_auc", ascending=False)

        print(f"\nEvaluated {len(results)} models:")
        print(df[["model_type", "architecture", "roc_auc", "ece", "size_mb"]].to_string(index=False))

        # Save results
        output_path = args.output or (TABLES_DIR / "evaluation_results.csv")
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")

        if args.json_output:
            json_path = pathlib.Path(args.json_output)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ JSON results saved to: {json_path}")
    else:
        print("\n⚠ No models found to evaluate!")
        print("  Run 'make train-teacher' to train teacher models first.")


if __name__ == "__main__":
    main()
