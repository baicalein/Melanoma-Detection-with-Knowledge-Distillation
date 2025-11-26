#!/usr/bin/env python
"""Run full KD ablation experiments.

This script runs the focused KD hyperparameter search:
- T ∈ {1, 2}
- α ∈ {0.5, 0.9}

Usage:
    python scripts/run_kd_ablation.py --teacher-ckpt models/checkpoints/teacher_best.pth
"""

import argparse
import json
import logging
import pathlib
import subprocess
import sys
from datetime import datetime
from itertools import product

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.config import FIGURES_DIR, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Focused hyperparameter search space per feedback
TEMPERATURES = [1.0, 2.0]
ALPHAS = [0.5, 0.9]


def run_experiment(
    teacher_ckpt: str,
    temperature: float,
    alpha: float,
    epochs: int = 50,
    seed: int = 42,
) -> dict:
    """Run a single KD experiment."""
    exp_name = f"student_T{temperature}_alpha{alpha}"
    logger.info(f"Running experiment: {exp_name}")

    cmd = [
        sys.executable,
        "scripts/train_student.py",
        "--teacher-ckpt",
        teacher_ckpt,
        "--temperature",
        str(temperature),
        "--alpha",
        str(alpha),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--name",
        exp_name,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Completed: {exp_name}")
        return {"name": exp_name, "success": True, "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {exp_name}")
        logger.error(e.stderr)
        return {"name": exp_name, "success": False, "error": e.stderr}


def collect_results() -> dict:
    """Collect results from all experiments."""
    results = {}

    for T, alpha in product(TEMPERATURES, ALPHAS):
        exp_name = f"student_T{T}_alpha{alpha}"
        metrics_path = FIGURES_DIR / "training" / exp_name / "student_holdout_metrics.json"

        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                results[exp_name] = json.load(f)
        else:
            logger.warning(f"Metrics not found for {exp_name}")

    return results


def print_summary(results: dict) -> None:
    """Print summary table of all experiments."""
    logger.info("=" * 80)
    logger.info("KD ABLATION RESULTS SUMMARY")
    logger.info("=" * 80)

    header = (
        f"{'Experiment':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10} {'ECE':>10} {'Spec@95%':>10}"
    )
    logger.info(header)
    logger.info("-" * 80)

    for name, metrics in sorted(results.items()):
        row = (
            f"{name:<25} "
            f"{metrics.get('roc_auc', 0):>10.4f} "
            f"{metrics.get('pr_auc', 0):>10.4f} "
            f"{metrics.get('f1', 0):>10.4f} "
            f"{metrics.get('ece', 0):>10.4f} "
            f"{metrics.get('specificity_at_target_sens', 0):>10.4f}"
        )
        logger.info(row)

    # Find best configuration
    if results:
        best_name = max(results.keys(), key=lambda k: results[k].get("roc_auc", 0))
        best_auc = results[best_name].get("roc_auc", 0)
        logger.info("-" * 80)
        logger.info(f"Best configuration: {best_name} (ROC-AUC = {best_auc:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Run KD ablation experiments")
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        required=True,
        help="Path to teacher checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-training", action="store_true", help="Only collect results")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("KD ABLATION EXPERIMENTS")
    logger.info(f"Teacher: {args.teacher_ckpt}")
    logger.info(f"Search space: T ∈ {TEMPERATURES}, α ∈ {ALPHAS}")
    logger.info(f"Total experiments: {len(TEMPERATURES) * len(ALPHAS)}")
    logger.info("=" * 80)

    if not args.skip_training:
        # Run all experiments
        experiment_results = []
        for T, alpha in product(TEMPERATURES, ALPHAS):
            result = run_experiment(
                args.teacher_ckpt,
                T,
                alpha,
                epochs=args.epochs,
                seed=args.seed,
            )
            experiment_results.append(result)

        # Save experiment log
        log_path = LOGS_DIR / f"kd_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2)
        logger.info(f"Experiment log saved to {log_path}")

    # Collect and summarize results
    results = collect_results()

    if results:
        print_summary(results)

        # Save summary
        summary_path = FIGURES_DIR / "kd_ablation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
    else:
        logger.warning("No results found. Run training first.")


if __name__ == "__main__":
    main()
