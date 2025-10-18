"""
Training utilities and helper functions.

This module contains common functions for training deep learning models,
including learning rate schedulers, metric tracking, and early stopping.
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)  # Ensures reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters

    Example:
        >>> model = nn.Linear(10, 5)
        >>> print(f"Parameters: {count_parameters(model):,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the device for training (CUDA, MPS, or CPU).

    Args:
        use_cuda: Whether to use CUDA if available

    Returns:
        torch.device object

    Example:
        >>> device = get_device()
        >>> print(device)  # cuda:0, mps, or cpu
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.

    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     loss_meter.update(loss.item(), batch_size)
        >>> print(f"Average loss: {loss_meter.avg:.4f}")
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics with new value.

        Args:
            val: New value to add
            n: Number of items represented by val
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        min_delta: Minimum change to qualify as improvement

    Example:
        >>> early_stop = EarlyStopping(patience=5, mode='min')
        >>> for epoch in range(num_epochs):
        ...     val_loss = validate(model, val_loader)
        ...     if early_stop(val_loss):
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(self, patience: int = 7, mode: str = "min", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

        if mode == "min":
            self.monitor_op = lambda x, y: x < y - min_delta
        else:
            self.monitor_op = lambda x, y: x > y + min_delta

    def __call__(self, metric: float) -> bool:
        """
        Check if early stopping criteria is met.

        Args:
            metric: Current metric value to check

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.monitor_op(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def save_checkpoint(
    state: dict[str, Any],
    filepath: str,
    is_best: bool = False,
) -> None:
    """
    Save model checkpoint and optionally save as best model.

    Args:
        state: Dictionary containing model state and metadata
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far

    Example:
        >>> checkpoint = {
        ...     "epoch": epoch,
        ...     "model_state_dict": model.state_dict(),
        ...     "optimizer_state_dict": optimizer.state_dict(),
        ...     "loss": loss,
        ... }
        >>> save_checkpoint(checkpoint, "checkpoint.pth", is_best=True)
    """
    torch.save(state, filepath)
    if is_best:
        best_path = filepath.replace(".pth", "_best.pth")
        torch.save(state, best_path)


def load_checkpoint(
    filepath: str, model: nn.Module, optimizer: Any | None = None
) -> dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Dictionary containing checkpoint metadata

    Example:
        >>> checkpoint = load_checkpoint("checkpoint.pth", model, optimizer)
        >>> start_epoch = checkpoint["epoch"] + 1
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
