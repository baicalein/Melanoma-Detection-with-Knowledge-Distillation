"""Evaluation metrics for melanoma detection.

Provides comprehensive metrics including:
- ROC-AUC, PR-AUC
- Sensitivity, Specificity, PPV, NPV at operating points
- Expected Calibration Error (ECE)
- Reliability diagrams
- Deployment metrics (size, latency, FLOPs)
"""

import json
import logging
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    accuracy: float
    precision: float
    recall: float  # Same as sensitivity
    f1: float
    specificity: float
    roc_auc: float
    pr_auc: float

    # Metrics at target sensitivity (e.g., 95%)
    threshold_at_target_sens: float
    specificity_at_target_sens: float
    ppv_at_target_sens: float
    npv_at_target_sens: float

    # Calibration
    ece: float
    mce: float  # Maximum calibration error

    # Sample counts
    n_samples: int
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class DeploymentMetrics:
    """Container for deployment-related metrics."""

    # Model size
    total_params: int
    trainable_params: int
    model_size_mb: float

    # Inference performance
    avg_latency_ms: float
    std_latency_ms: float
    throughput_images_per_sec: float

    # Quantization impact (if applicable)
    quantized_size_mb: float | None = None
    quantized_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.95,
    ece_bins: int = 15,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth binary labels (N,)
        y_prob: Predicted probabilities for positive class (N,)
        target_sensitivity: Target sensitivity for threshold selection
        ece_bins: Number of bins for calibration error

    Returns:
        ClassificationMetrics dataclass

    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Default threshold predictions
    y_pred = (y_prob >= 0.5).astype(int)

    # Basic metrics at 0.5 threshold
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Compute confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    # PR-AUC
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec_curve, prec_curve)
    except ValueError:
        pr_auc = 0.0

    # Find threshold for target sensitivity
    threshold, spec_at_sens, ppv_at_sens, npv_at_sens = find_threshold_at_sensitivity(
        y_true, y_prob, target_sensitivity
    )

    # Calibration metrics
    ece, mce = compute_calibration_error(y_true, y_prob, n_bins=ece_bins)

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        specificity=float(specificity),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        threshold_at_target_sens=float(threshold),
        specificity_at_target_sens=float(spec_at_sens),
        ppv_at_target_sens=float(ppv_at_sens),
        npv_at_target_sens=float(npv_at_sens),
        ece=float(ece),
        mce=float(mce),
        n_samples=len(y_true),
        n_positive=int(y_true.sum()),
        n_negative=int((1 - y_true).sum()),
    )


def find_threshold_at_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.95,
) -> tuple[float, float, float, float]:
    """Find threshold that achieves target sensitivity.

    Returns:
        Tuple of (threshold, specificity, ppv, npv) at that threshold

    """
    # Get all unique thresholds from ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Find threshold where TPR (sensitivity) >= target
    # TPR = tpr, we want tpr >= target_sensitivity
    valid_indices = np.where(tpr >= target_sensitivity)[0]

    if len(valid_indices) == 0:
        # Can't achieve target sensitivity, use lowest threshold
        idx = 0
    else:
        # Choose the threshold with highest specificity among those meeting sensitivity
        # Higher threshold = higher specificity
        idx = valid_indices[-1]  # Highest threshold meeting sensitivity

    threshold = thresholds[idx] if idx < len(thresholds) else 0.5

    # Compute metrics at this threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return float(threshold), float(specificity), float(ppv), float(npv)


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Tuple of (ECE, MCE)

    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])

        if in_bin.sum() == 0:
            continue

        # Accuracy in this bin (fraction of correct predictions)
        bin_accuracy = y_true[in_bin].mean()

        # Average confidence in this bin
        bin_confidence = y_prob[in_bin].mean()

        # Calibration error for this bin
        bin_error = abs(bin_accuracy - bin_confidence)

        # Weighted contribution to ECE
        ece += (in_bin.sum() / n_samples) * bin_error

        # Update MCE
        mce = max(mce, bin_error)

    return ece, mce


def get_reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get data for reliability diagram.

    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_counts)

    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        bin_counts[i] = in_bin.sum()

        if in_bin.sum() > 0:
            bin_accuracies[i] = y_true[in_bin].mean()

    return bin_centers, bin_accuracies, bin_counts


def compute_deployment_metrics(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
) -> DeploymentMetrics:
    """Compute deployment-related metrics: size, latency, throughput.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, height, width)
        device: Device for inference benchmarking
        warmup_iterations: Warmup iterations before timing
        benchmark_iterations: Number of iterations for timing

    Returns:
        DeploymentMetrics dataclass

    """
    model = model.to(device)
    model.eval()

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model size
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_bytes + buffer_bytes) / (1024**2)

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_iterations):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    avg_latency = latencies.mean()
    std_latency = latencies.std()

    # Throughput (images/sec)
    batch_size = input_shape[0]
    throughput = (batch_size * 1000) / avg_latency  # Convert from ms to sec

    return DeploymentMetrics(
        total_params=total_params,
        trainable_params=trainable_params,
        model_size_mb=model_size_mb,
        avg_latency_ms=float(avg_latency),
        std_latency_ms=float(std_latency),
        throughput_images_per_sec=float(throughput),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    target_sensitivity: float = 0.95,
    ece_bins: int = 15,
) -> ClassificationMetrics:
    """Evaluate model on a dataloader and compute all metrics.

    Args:
        model: PyTorch model
        dataloader: DataLoader with (images, targets)
        device: Device for inference
        target_sensitivity: Target sensitivity for threshold
        ece_bins: Number of bins for ECE

    Returns:
        ClassificationMetrics

    """
    model = model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    for images, targets in dataloader:
        images = images.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_targets.append(targets.numpy())
        all_probs.append(probs)

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    return compute_classification_metrics(y_true, y_prob, target_sensitivity, ece_bins)


def compare_models(
    results: dict[str, ClassificationMetrics],
    baseline_name: str = "teacher",
) -> dict[str, dict[str, float]]:
    """Compare model metrics, computing differences from baseline.

    Args:
        results: Dict mapping model name to ClassificationMetrics
        baseline_name: Name of baseline model for comparison

    Returns:
        Dict with differences (Δ) for each metric

    """
    if baseline_name not in results:
        raise ValueError(f"Baseline '{baseline_name}' not in results")

    baseline = results[baseline_name]
    comparisons = {}

    for name, metrics in results.items():
        if name == baseline_name:
            continue

        delta = {
            "Δ_roc_auc": metrics.roc_auc - baseline.roc_auc,
            "Δ_pr_auc": metrics.pr_auc - baseline.pr_auc,
            "Δ_specificity_at_95sens": metrics.specificity_at_target_sens
            - baseline.specificity_at_target_sens,
            "Δ_ece": metrics.ece - baseline.ece,
        }
        comparisons[name] = delta

    return comparisons
