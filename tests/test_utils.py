"""
Tests for utility functions.
"""

import pytest
import torch

from deep_learning_final_project.utils.metrics import (
    MetricTracker,
    compute_accuracy,
    compute_top_k_accuracy,
)
from deep_learning_final_project.utils.training import (
    AverageMeter,
    EarlyStopping,
    count_parameters,
    get_device,
    set_seed,
)


class TestTrainingUtils:
    """Tests for training utilities."""

    def test_set_seed(self) -> None:
        """Test that setting seed produces reproducible results."""
        set_seed(42)
        x1 = torch.randn(10)

        set_seed(42)
        x2 = torch.randn(10)

        assert torch.allclose(x1, x2)

    def test_get_device(self) -> None:
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_count_parameters(self) -> None:
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        count = count_parameters(model)
        # 10 * 5 weights + 5 biases = 55 parameters
        assert count == 55

    def test_average_meter(self) -> None:
        """Test AverageMeter functionality."""
        meter = AverageMeter()

        meter.update(10.0, n=1)
        meter.update(20.0, n=1)

        assert meter.avg == 15.0
        assert meter.count == 2

    def test_average_meter_reset(self) -> None:
        """Test AverageMeter reset."""
        meter = AverageMeter()
        meter.update(10.0)

        meter.reset()
        assert meter.avg == 0.0
        assert meter.count == 0

    def test_early_stopping_min_mode(self) -> None:
        """Test early stopping in min mode (for loss)."""
        early_stop = EarlyStopping(patience=2, mode="min")

        # Improving
        assert not early_stop(1.0)
        assert not early_stop(0.9)
        assert not early_stop(0.8)

        # Not improving for patience iterations
        assert not early_stop(0.9)  # Counter = 1, not stopped yet
        assert early_stop(0.9)  # Counter = 2, now triggers stop

    def test_early_stopping_max_mode(self) -> None:
        """Test early stopping in max mode (for accuracy)."""
        early_stop = EarlyStopping(patience=2, mode="max")

        # Improving
        assert not early_stop(0.8)
        assert not early_stop(0.85)
        assert not early_stop(0.9)

        # Not improving for patience iterations
        assert not early_stop(0.89)  # Counter = 1, not stopped yet
        assert early_stop(0.88)  # Counter = 2, now triggers stop


class TestMetrics:
    """Tests for metric functions."""

    def test_compute_accuracy(self) -> None:
        """Test accuracy computation."""
        predictions = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 1, 1])

        accuracy = compute_accuracy(predictions, targets)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_compute_accuracy_with_logits(self) -> None:
        """Test accuracy computation with logits."""
        logits = torch.tensor([
            [2.0, 0.5, 0.1],  # Predicts class 0
            [0.1, 2.0, 0.5],  # Predicts class 1
        ])
        targets = torch.tensor([0, 1])

        accuracy = compute_accuracy(logits, targets)
        assert accuracy == 1.0

    def test_compute_top_k_accuracy(self) -> None:
        """Test top-k accuracy computation."""
        logits = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],  # Top 1: 4, Top 2: [4, 3]
            [5.0, 4.0, 3.0, 2.0, 1.0],  # Top 1: 0, Top 2: [0, 1]
        ])
        targets = torch.tensor([3, 1])  # True classes

        top1_acc = compute_top_k_accuracy(logits, targets, k=1)
        top2_acc = compute_top_k_accuracy(logits, targets, k=2)

        assert top1_acc == 0.0  # Neither prediction is correct in top-1
        assert top2_acc == 1.0  # Both are correct in top-2

    def test_metric_tracker(self) -> None:
        """Test MetricTracker functionality."""
        tracker = MetricTracker()

        tracker.update({"loss": 1.0, "acc": 0.8}, n=10)
        tracker.update({"loss": 0.8, "acc": 0.9}, n=10)

        avg = tracker.get_average()
        assert avg["loss"] == 0.9
        assert avg["acc"] == 0.85

    def test_metric_tracker_reset(self) -> None:
        """Test MetricTracker reset."""
        tracker = MetricTracker()
        tracker.update({"loss": 1.0})

        tracker.reset()
        assert tracker.metrics == {}
        assert tracker.counts == {}
