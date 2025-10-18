"""
Base model class and utilities.

This module provides a base class for all models in the project, with common
functionality like saving, loading, and parameter counting.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model class with common functionality.

    All models in this project should inherit from this class to ensure
    consistency and reusability.

    Example:
        >>> class MyModel(BaseModel):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layer = nn.Linear(10, 5)
        ...
        ...     def forward(self, x):
        ...         return self.layer(x)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters

        Example:
            >>> model = MyModel()
            >>> print(f"Parameters: {model.count_parameters():,}")
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimizer_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: State dict of optimizer (optional)
            metrics: Dictionary of metrics to save (optional)

        Example:
            >>> model.save_checkpoint(
            ...     "checkpoints/model_epoch_10.pth",
            ...     epoch=10,
            ...     metrics={"val_loss": 0.5}
            ... )
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        if metrics is not None:
            checkpoint["metrics"] = metrics

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str | Path,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load checkpoint to

        Returns:
            Dictionary containing checkpoint metadata

        Example:
            >>> model = MyModel()
            >>> checkpoint_info = model.load_checkpoint("checkpoints/best.pth")
            >>> print(f"Loaded epoch: {checkpoint_info['epoch']}")
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.

        Returns:
            String representation of model summary
        """
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)

        summary = [
            f"Model: {self.__class__.__name__}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {total_params - trainable_params:,}",
        ]

        return "\n".join(summary)

    def freeze_layers(self, layer_names: Optional[list[str]] = None) -> None:
        """
        Freeze specified layers or all layers if none specified.

        Args:
            layer_names: List of layer names to freeze. If None, freeze all layers.

        Example:
            >>> model.freeze_layers(['conv1', 'bn1'])
        """
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False

    def unfreeze_layers(self, layer_names: Optional[list[str]] = None) -> None:
        """
        Unfreeze specified layers or all layers if none specified.

        Args:
            layer_names: List of layer names to unfreeze. If None, unfreeze all.

        Example:
            >>> model.unfreeze_layers(['fc'])
        """
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
