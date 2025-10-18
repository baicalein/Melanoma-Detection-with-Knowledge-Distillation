"""
Evaluation metrics for model performance.

This module provides common metrics for evaluating deep learning models,
including classification and regression metrics.
"""

from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels

    Returns:
        Accuracy as a float between 0 and 1

    Example:
        >>> preds = torch.tensor([0, 1, 2, 1])
        >>> targets = torch.tensor([0, 1, 1, 1])
        >>> acc = compute_accuracy(preds, targets)
        >>> print(f"Accuracy: {acc:.2%}")
    """
    if predictions.ndim > 1:
        predictions = predictions.argmax(dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def compute_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute top-k accuracy.

    Args:
        predictions: Model predictions (logits)
        targets: Ground truth labels
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as a float

    Example:
        >>> logits = torch.randn(10, 100)  # 10 samples, 100 classes
        >>> targets = torch.randint(0, 100, (10,))
        >>> top5_acc = compute_top_k_accuracy(logits, targets, k=5)
    """
    with torch.no_grad():
        _, top_k_preds = predictions.topk(k, dim=1, largest=True, sorted=True)
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=1).sum().item()
        return correct / targets.size(0)


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = "macro"
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: Predicted class labels
        targets: True class labels
        average: Averaging strategy for multi-class ('macro', 'micro', 'weighted')

    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score

    Example:
        >>> preds = np.array([0, 1, 2, 1])
        >>> targets = np.array([0, 1, 1, 1])
        >>> metrics = compute_classification_metrics(preds, targets)
        >>> print(f"F1 Score: {metrics['f1_score']:.3f}")
    """
    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, average=average, zero_division=0),
        "recall": recall_score(targets, predictions, average=average, zero_division=0),
        "f1_score": f1_score(targets, predictions, average=average, zero_division=0),
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class labels
        targets: True class labels

    Returns:
        Confusion matrix as numpy array

    Example:
        >>> preds = np.array([0, 1, 2, 1])
        >>> targets = np.array([0, 1, 1, 1])
        >>> cm = compute_confusion_matrix(preds, targets)
    """
    return confusion_matrix(targets, predictions)


def compute_per_class_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> List[float]:
    """
    Compute accuracy for each class separately.

    Args:
        predictions: Predicted class labels
        targets: True class labels
        num_classes: Total number of classes

    Returns:
        List of per-class accuracies

    Example:
        >>> preds = np.array([0, 1, 2, 1, 0])
        >>> targets = np.array([0, 1, 1, 1, 0])
        >>> per_class_acc = compute_per_class_accuracy(preds, targets, num_classes=3)
    """
    accuracies = []
    for class_idx in range(num_classes):
        mask = targets == class_idx
        if mask.sum() > 0:
            class_acc = (predictions[mask] == targets[mask]).sum() / mask.sum()
            accuracies.append(float(class_acc))
        else:
            accuracies.append(0.0)
    return accuracies


class MetricTracker:
    """
    Track multiple metrics during training and validation.

    Example:
        >>> tracker = MetricTracker()
        >>> tracker.update({"loss": 0.5, "accuracy": 0.85})
        >>> tracker.update({"loss": 0.4, "accuracy": 0.87})
        >>> print(tracker.get_average())  # {"loss": 0.45, "accuracy": 0.86}
    """

    def __init__(self) -> None:
        self.metrics: dict = {}
        self.counts: dict = {}

    def update(self, metrics: dict, n: int = 1) -> None:
        """
        Update tracked metrics.

        Args:
            metrics: Dictionary of metric names and values
            n: Number of samples represented by these metrics
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value * n
            self.counts[key] += n

    def get_average(self) -> dict:
        """Get average values for all tracked metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics = {}
        self.counts = {}

    def get_current(self) -> dict:
        """Get current metric values without averaging."""
        return self.metrics.copy()
