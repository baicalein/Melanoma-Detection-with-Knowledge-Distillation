"""
Training visualization and reliability diagrams.

Provides:
- Training curves (loss, accuracy, ROC-AUC)
- Reliability diagrams for calibration
- ROC and PR curves
- Model comparison plots
"""

import pathlib
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from src.config import FIGURES_DIR
from src.evaluation.metrics import get_reliability_diagram_data


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Dict with train_loss, val_loss, train_acc, val_acc, val_roc_auc
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    # Loss
    ax = axes[0]
    ax.plot(epochs, history.get("train_loss", []), "b-", label="Train", linewidth=2)
    ax.plot(epochs, history.get("val_loss", []), "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history.get("train_acc", []), "b-", label="Train", linewidth=2)
    ax.plot(epochs, history.get("val_acc", []), "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROC-AUC
    ax = axes[2]
    ax.plot(epochs, history.get("val_roc_auc", []), "g-", label="Val ROC-AUC", linewidth=2)
    ax.plot(epochs, history.get("val_f1", []), "m-", label="Val F1", linewidth=2)
    
    # Mark best epoch
    if "val_roc_auc" in history and history["val_roc_auc"]:
        best_idx = np.argmax(history["val_roc_auc"])
        best_auc = history["val_roc_auc"][best_idx]
        ax.axvline(x=best_idx + 1, color="k", linestyle="--", alpha=0.5)
        ax.annotate(
            f"Best: {best_auc:.4f}",
            xy=(best_idx + 1, best_auc),
            xytext=(5, -20),
            textcoords="offset points",
            fontsize=10,
        )
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """
    Plot reliability diagram for calibration assessment.
    
    A well-calibrated model should have bars close to the diagonal.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    bin_centers, bin_accuracies, bin_counts = get_reliability_diagram_data(
        y_true, y_prob, n_bins
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax = axes[0]
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
    
    # Bar chart of actual vs predicted
    bar_width = 0.8 / n_bins
    ax.bar(
        bin_centers,
        bin_accuracies,
        width=bar_width,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Model",
    )
    
    # Gap visualization
    for i, (center, acc) in enumerate(zip(bin_centers, bin_accuracies)):
        if bin_counts[i] > 0:
            gap = acc - center
            color = "green" if gap >= 0 else "red"
            ax.plot([center, center], [center, acc], color=color, linewidth=2, alpha=0.6)
    
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax = axes[1]
    ax.bar(
        bin_centers,
        bin_counts / bin_counts.sum(),
        width=bar_width,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Samples", fontsize=12)
    ax.set_title("Prediction Distribution", fontsize=14)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC and PR Curves",
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """
    Plot ROC and Precision-Recall curves.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    ax = axes[0]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = np.trapz(tpr, fpr)
    
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    
    # Mark 95% sensitivity point
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    ax.scatter([fpr[idx_95]], [tpr[idx_95]], color="red", s=100, zorder=5)
    ax.annotate(
        f"@95% sens\nSpec={1-fpr[idx_95]:.2f}",
        xy=(fpr[idx_95], tpr[idx_95]),
        xytext=(20, -20),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red"),
    )
    
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # PR curve
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(precision, recall)
    
    # Baseline (prevalence)
    prevalence = y_true.mean()
    ax.axhline(y=prevalence, color="k", linestyle="--", linewidth=1, label=f"Baseline = {prevalence:.2f}")
    
    ax.plot(recall, precision, "b-", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["roc_auc", "pr_auc", "ece", "specificity_at_95sens"],
    title: str = "Model Comparison",
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing multiple models on key metrics.
    
    Args:
        results: Dict mapping model name to metrics dict
        metrics: List of metric names to plot
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    n_models = len(results)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    model_names = list(results.keys())
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[m].get(metric, 0) for m in model_names]
        
        bars = ax.bar(range(n_models), values, color=colors)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_deployment_metrics(
    fp32_metrics: Dict[str, float],
    int8_metrics: Dict[str, float],
    title: str = "Deployment Metrics: FP32 vs INT8",
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """
    Compare FP32 and INT8 models on deployment metrics.
    
    Args:
        fp32_metrics: Dict with size_mb, latency_ms, roc_auc
        int8_metrics: Dict with size_mb, latency_ms, roc_auc
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Model size
    ax = axes[0]
    sizes = [fp32_metrics["size_mb"], int8_metrics["size_mb"]]
    bars = ax.bar(["FP32", "INT8"], sizes, color=["#1f77b4", "#ff7f0e"])
    for bar, size in zip(bars, sizes):
        ax.annotate(
            f"{size:.1f} MB",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
        )
    ax.set_ylabel("Size (MB)")
    ax.set_title("Model Size")
    reduction = 1 - (int8_metrics["size_mb"] / fp32_metrics["size_mb"])
    ax.text(0.5, 0.95, f"{reduction:.0%} reduction", transform=ax.transAxes, ha="center")
    
    # Latency
    ax = axes[1]
    latencies = [fp32_metrics["latency_ms"], int8_metrics["latency_ms"]]
    bars = ax.bar(["FP32", "INT8"], latencies, color=["#1f77b4", "#ff7f0e"])
    for bar, lat in zip(bars, latencies):
        ax.annotate(
            f"{lat:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
        )
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency")
    speedup = fp32_metrics["latency_ms"] / int8_metrics["latency_ms"]
    ax.text(0.5, 0.95, f"{speedup:.1f}x speedup", transform=ax.transAxes, ha="center")
    
    # ROC-AUC
    ax = axes[2]
    aucs = [fp32_metrics["roc_auc"], int8_metrics["roc_auc"]]
    bars = ax.bar(["FP32", "INT8"], aucs, color=["#1f77b4", "#ff7f0e"])
    for bar, auc_val in zip(bars, aucs):
        ax.annotate(
            f"{auc_val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
        )
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC")
    ax.set_ylim([0.8, 1.0])  # Zoom in for better visibility
    delta = int8_metrics["roc_auc"] - fp32_metrics["roc_auc"]
    ax.text(0.5, 0.95, f"Δ = {delta:+.4f}", transform=ax.transAxes, ha="center")
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig
