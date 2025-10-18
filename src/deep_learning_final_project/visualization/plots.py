"""
Plotting utilities for visualizing results and metrics.

This module provides functions for creating common plots used in deep learning
research, including training curves, confusion matrices, and sample visualizations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    metric_name: str = "Loss",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training metric values
        val_losses: List of validation metric values
        metric_name: Name of the metric being plotted
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> train_losses = [0.8, 0.6, 0.4, 0.3]
        >>> val_losses = [0.9, 0.7, 0.5, 0.4]
        >>> fig = plot_training_curves(train_losses, val_losses)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label=f"Train {metric_name}", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label=f"Val {metric_name}", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Training and Validation {metric_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str] | None = None,
    normalize: bool = False,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: Optional list of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> cm = np.array([[50, 2], [3, 45]])
        >>> fig = plot_confusion_matrix(cm, class_names=['Class A', 'Class B'])
    """
    if normalize:
        cm_sum = confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = confusion_matrix.astype("float") / cm_sum

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count" if not normalize else "Proportion"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_sample_predictions(
    images: torch.Tensor,
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str] | None = None,
    num_samples: int = 16,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 12),
) -> Figure:
    """
    Plot sample images with true and predicted labels.

    Args:
        images: Tensor of images [N, C, H, W]
        true_labels: List of true labels
        pred_labels: List of predicted labels
        class_names: Optional list of class names
        num_samples: Number of samples to display
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    num_samples = min(num_samples, len(images))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx in range(num_samples):
        ax = axes[idx]

        # Convert image to numpy and transpose for plotting
        img = images[idx].cpu().numpy()
        if img.shape[0] == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[0] == 1:  # Grayscale
            img = img.squeeze()

        # Denormalize if needed (assuming ImageNet normalization)
        img = (img - img.min()) / (img.max() - img.min())

        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.axis("off")

        true_name = class_names[true_labels[idx]] if class_names else true_labels[idx]
        pred_name = class_names[pred_labels[idx]] if class_names else pred_labels[idx]

        color = "green" if true_labels[idx] == pred_labels[idx] else "red"
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=10, color=color)

    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_learning_rate_schedule(
    learning_rates: list[float],
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot learning rate schedule over training.

    Args:
        learning_rates: List of learning rates per epoch
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(learning_rates) + 1)
    ax.plot(epochs, learning_rates, "b-", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def setup_publication_style() -> None:
    """
    Set up matplotlib style for publication-quality figures.

    Call this function at the start of your visualization scripts.

    Example:
        >>> setup_publication_style()
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> plt.show()
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "text.usetex": False,  # Set to True if you have LaTeX installed
        }
    )
