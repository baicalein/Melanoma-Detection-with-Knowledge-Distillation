"""Quantize student model and evaluate.

Usage:
    python scripts/quantize_model.py --model-ckpt models/checkpoints/student_T2_alpha0.5_best.pth
"""

import argparse
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import torch

from src.config import (
    FIGURES_DIR,
    LOGS_DIR,
    DataConfig,
    StudentConfig,
    set_seed,
)
from src.data.dataset import HAM10000Dataset, get_eval_transforms
from src.data.splits import load_or_create_splits
from src.evaluation.quantization import (
    compare_quantized_model,
    get_model_size_mb,
    quantize_model_dynamic,
    quantize_model_static,
    save_quantized_model,
)
from src.models.architectures import StudentModel
from src.plotting.training_plots import plot_deployment_metrics, plot_reliability_diagram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "quantize.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize student model")
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Path to student model checkpoint",
    )
    parser.add_argument(
        "--student-arch",
        type=str,
        default="mobilenet_v3_small",
        help="Student architecture",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Quantization method",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_name = args.name or f"quantized_{args.method}"
    logger.info(f"Experiment: {exp_name}")

    # Load data
    data_config = DataConfig()
    _, val_path, holdout_path = load_or_create_splits()

    # Create dataloaders
    holdout_dataset = HAM10000Dataset(
        holdout_path,
        transform=get_eval_transforms(data_config),
    )
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
    )

    val_dataset = HAM10000Dataset(
        val_path,
        transform=get_eval_transforms(data_config),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
    )

    # Load FP32 student
    logger.info(f"Loading student from {args.model_ckpt}...")
    student_config = StudentConfig(architecture=args.student_arch)
    student_fp32 = StudentModel(student_config)

    checkpoint = torch.load(args.model_ckpt, map_location="cpu")
    if "model_state_dict" in checkpoint:
        student_fp32.load_state_dict(checkpoint["model_state_dict"])
    else:
        student_fp32.load_state_dict(checkpoint)

    student_fp32.eval()

    fp32_size = get_model_size_mb(student_fp32)
    logger.info(f"FP32 model size: {fp32_size:.2f} MB")

    # Quantize
    logger.info(f"Applying {args.method} quantization...")
    if args.method == "dynamic":
        student_int8 = quantize_model_dynamic(student_fp32)
    else:
        # Static quantization with calibration
        student_int8 = quantize_model_static(student_fp32, val_loader)

    int8_size = get_model_size_mb(student_int8)
    logger.info(f"INT8 model size: {int8_size:.2f} MB")
    logger.info(f"Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")

    # Compare models
    logger.info("Comparing FP32 vs INT8...")
    comparison = compare_quantized_model(student_fp32, student_int8, holdout_loader, device="cpu")

    # Log results
    logger.info("=" * 60)
    logger.info("QUANTIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<25} {'FP32':>12} {'INT8':>12} {'Δ':>10}")
    logger.info("-" * 60)
    logger.info(
        f"{'Size (MB)':<25} {comparison['fp32']['size_mb']:>12.2f} {comparison['int8']['size_mb']:>12.2f} {-comparison['delta']['size_reduction'] * 100:>+9.1f}%"
    )
    logger.info(
        f"{'Latency (ms)':<25} {comparison['fp32']['latency_ms']:>12.2f} {comparison['int8']['latency_ms']:>12.2f} {comparison['delta']['latency_speedup']:>9.2f}x"
    )
    logger.info(
        f"{'ROC-AUC':<25} {comparison['fp32']['roc_auc']:>12.4f} {comparison['int8']['roc_auc']:>12.4f} {comparison['delta']['delta_roc_auc']:>+10.4f}"
    )
    logger.info(
        f"{'ECE':<25} {comparison['fp32']['ece']:>12.4f} {comparison['int8']['ece']:>12.4f} {comparison['delta']['delta_ece']:>+10.4f}"
    )

    # Check if degradation is acceptable
    auc_drop = abs(comparison["delta"]["delta_roc_auc"])
    ece_increase = comparison["delta"]["delta_ece"]

    if auc_drop > 0.02:
        logger.warning(f"ROC-AUC dropped by {auc_drop:.4f} (>0.02). Consider QAT.")
    else:
        logger.info(f"ROC-AUC drop acceptable: {auc_drop:.4f} (≤0.02)")

    if ece_increase > 0.05:
        logger.warning(f"ECE increased by {ece_increase:.4f} (>0.05). Calibration degraded.")
    else:
        logger.info(f"ECE increase acceptable: {ece_increase:.4f} (≤0.05)")

    # Save quantized model
    save_path = save_quantized_model(student_int8, exp_name)

    # Save comparison results
    fig_dir = FIGURES_DIR / "quantization" / exp_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(fig_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Plot comparison
    plot_deployment_metrics(
        comparison["fp32"],
        comparison["int8"],
        title="FP32 vs INT8 Quantization",
        save_path=fig_dir / "deployment_comparison.png",
    )

    # Reliability diagrams for both
    def get_preds(model, loader):
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.cpu())
                probs = torch.sigmoid(logits).numpy()
                all_probs.append(probs)
                all_targets.append(targets.numpy())
        return np.concatenate(all_targets), np.concatenate(all_probs)

    y_true, int8_prob = get_preds(student_int8, holdout_loader)

    plot_reliability_diagram(
        y_true,
        int8_prob,
        title="INT8 Quantized Model Calibration",
        save_path=fig_dir / "reliability_diagram_int8.png",
    )

    # Final summary
    logger.info("=" * 60)
    logger.info("DEPLOYMENT READINESS")
    logger.info("=" * 60)

    mobile_target = 25.0  # MB
    edge_target = 2.0  # MB

    meets_mobile = int8_size <= mobile_target
    meets_edge = int8_size <= edge_target

    logger.info(
        f"Mobile target (<{mobile_target} MB): {'✓' if meets_mobile else '✗'} ({int8_size:.2f} MB)"
    )
    logger.info(
        f"Edge target (<{edge_target} MB): {'✓' if meets_edge else '✗'} ({int8_size:.2f} MB)"
    )

    logger.info("Quantization complete!")


if __name__ == "__main__":
    main()
