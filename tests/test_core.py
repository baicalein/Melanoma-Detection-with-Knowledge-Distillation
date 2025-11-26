"""Unit tests for core functionality.

Run with: pytest tests/ -v
"""

import pathlib
import sys

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


class TestConfig:
    """Test configuration module."""

    def test_imports(self):
        from src.config import (
            ExperimentConfig,
        )

        config = ExperimentConfig()
        assert config.seed == 42
        assert config.data.batch_size == 32

    def test_set_seed(self):
        from src.config import set_seed

        set_seed(42)
        a = torch.rand(10)

        set_seed(42)
        b = torch.rand(10)

        assert torch.allclose(a, b)


class TestModels:
    """Test model architectures."""

    def test_teacher_model(self):
        from src.config import TeacherConfig
        from src.models.architectures import TeacherModel

        config = TeacherConfig(architecture="resnet18")
        model = TeacherModel(config)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        out = model(x)

        assert out.shape == (2,)  # Binary classification
        assert model.count_parameters()["total"] > 0

    def test_student_model(self):
        from src.config import StudentConfig
        from src.models.architectures import StudentModel

        config = StudentConfig(architecture="mobilenet_v3_small")
        model = StudentModel(config)

        x = torch.randn(2, 3, 224, 224)
        out = model(x)

        assert out.shape == (2,)

        # Check deployment constraints
        constraints = model.check_deployment_constraints()
        assert "size_mb" in constraints
        assert constraints["size_mb"] < 25  # Should be under mobile target


class TestKDLoss:
    """Test knowledge distillation loss functions."""

    def test_kd_loss(self):
        from src.models.kd_loss import KnowledgeDistillationLoss

        loss_fn = KnowledgeDistillationLoss(temperature=2.0, alpha=0.5)

        student_logits = torch.randn(4)
        teacher_logits = torch.randn(4)
        targets = torch.tensor([0, 1, 1, 0], dtype=torch.float)

        loss_dict = loss_fn(student_logits, teacher_logits, targets)

        assert "loss" in loss_dict
        assert "soft_loss" in loss_dict
        assert "hard_loss" in loss_dict
        assert loss_dict["loss"].item() >= 0

    def test_focal_loss(self):
        from src.models.kd_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

        logits = torch.randn(4)
        targets = torch.tensor([0, 1, 1, 0], dtype=torch.float)

        loss = loss_fn(logits, targets)

        assert loss.item() >= 0


class TestMetrics:
    """Test evaluation metrics."""

    def test_classification_metrics(self):
        from src.evaluation.metrics import compute_classification_metrics

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.6])

        metrics = compute_classification_metrics(y_true, y_prob)

        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.pr_auc <= 1
        assert 0 <= metrics.ece <= 1
        assert metrics.n_samples == 8
        assert metrics.n_positive == 4

    def test_calibration_error(self):
        from src.evaluation.metrics import compute_calibration_error

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.6])

        ece, mce = compute_calibration_error(y_true, y_prob)

        assert 0 <= ece <= 1
        assert 0 <= mce <= 1
        assert ece <= mce  # MCE >= ECE by definition


class TestDataSplits:
    """Test data splitting utilities."""

    def test_lesion_split_no_leakage(self):
        """Verify lesion-level splitting prevents data leakage."""
        import pandas as pd

        from src.data.splits import create_lesion_level_splits

        # Create mock dataset with multiple images per lesion
        df = pd.DataFrame(
            {
                "lesion_id": [
                    "L1",
                    "L1",
                    "L2",
                    "L2",
                    "L3",
                    "L3",
                    "L4",
                    "L4",
                    "L5",
                    "L5",
                    "L6",
                    "L7",
                    "L8",
                    "L9",
                    "L10",
                    "L11",
                    "L12",
                    "L13",
                    "L14",
                    "L15",
                ],
                "image_id": [f"img_{i}" for i in range(20)],
                "target": [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            }
        )

        train_df, val_df, holdout_df = create_lesion_level_splits(
            df, train_ratio=0.7, val_ratio=0.15, holdout_ratio=0.15, random_seed=42
        )

        # Verify no lesion overlap
        train_lesions = set(train_df["lesion_id"])
        val_lesions = set(val_df["lesion_id"])
        holdout_lesions = set(holdout_df["lesion_id"])

        assert len(train_lesions & val_lesions) == 0
        assert len(train_lesions & holdout_lesions) == 0
        assert len(val_lesions & holdout_lesions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
