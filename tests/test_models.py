"""
Tests for model architectures.
"""

import torch

from deep_learning_final_project.models.base_model import BaseModel
from deep_learning_final_project.models.example_model import ResidualBlock, SimpleCNN


class TestSimpleCNN:
    """Tests for SimpleCNN model."""

    def test_model_creation(self) -> None:
        """Test that model can be created."""
        model = SimpleCNN(num_classes=10, in_channels=3)
        assert isinstance(model, BaseModel)

    def test_forward_pass(self) -> None:
        """Test forward pass with dummy input."""
        model = SimpleCNN(num_classes=10, in_channels=3)
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        assert output.shape == (2, 10)

    def test_parameter_count(self) -> None:
        """Test parameter counting."""
        model = SimpleCNN(num_classes=10, in_channels=3)
        param_count = model.count_parameters()
        assert param_count > 0

    def test_different_input_channels(self) -> None:
        """Test model with different input channels."""
        model = SimpleCNN(num_classes=5, in_channels=1)
        x = torch.randn(4, 1, 32, 32)
        output = model(x)
        assert output.shape == (4, 5)


class TestResidualBlock:
    """Tests for ResidualBlock."""

    def test_residual_block_creation(self) -> None:
        """Test that residual block can be created."""
        block = ResidualBlock(64, 64)
        assert isinstance(block, torch.nn.Module)

    def test_residual_block_forward(self) -> None:
        """Test forward pass through residual block."""
        block = ResidualBlock(64, 64)
        x = torch.randn(2, 64, 16, 16)
        output = block(x)
        assert output.shape == x.shape

    def test_residual_block_downsample(self) -> None:
        """Test residual block with stride > 1."""
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 16, 16)
        output = block(x)
        assert output.shape == (2, 128, 8, 8)


class TestBaseModel:
    """Tests for BaseModel functionality."""

    def test_freeze_layers(self) -> None:
        """Test layer freezing functionality."""
        model = SimpleCNN(num_classes=10)

        # Freeze all layers
        model.freeze_layers()
        for param in model.parameters():
            assert not param.requires_grad

    def test_unfreeze_layers(self) -> None:
        """Test layer unfreezing functionality."""
        model = SimpleCNN(num_classes=10)

        # Freeze then unfreeze
        model.freeze_layers()
        model.unfreeze_layers()
        for param in model.parameters():
            assert param.requires_grad

    def test_model_summary(self) -> None:
        """Test model summary generation."""
        model = SimpleCNN(num_classes=10)
        summary = model.get_model_summary()
        assert "SimpleCNN" in summary
        assert "parameters" in summary.lower()
