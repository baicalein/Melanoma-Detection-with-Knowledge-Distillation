"""Model inference utilities for prediction and visualization.

Provides functions for:
- Loading trained models from checkpoints
- Running inference on datasets
- Prediction visualization
- Confidence analysis
"""

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image


def load_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: pathlib.Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load model weights from checkpoint.

    Args:
        model: Initialized model architecture
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto

    Returns:
        Model with loaded weights in eval mode.

    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def get_prediction(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: str | torch.device,
) -> tuple[int, float]:
    """Get model prediction for a single image.

    Args:
        model: Trained model in eval mode
        image_tensor: Preprocessed image tensor (C, H, W)
        device: Device to run inference on

    Returns:
        Tuple of (prediction, probability).

    """
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logit = model(image_tensor)
        prob = torch.sigmoid(logit).item()
        pred = int(prob > 0.5)
    return pred, prob


def get_all_predictions(
    teacher_model: nn.Module,
    student_model: nn.Module,
    dataset: Any,
    device: str | torch.device,
) -> pd.DataFrame:
    """Get predictions from both teacher and student for all samples.

    Args:
        teacher_model: Teacher model in eval mode
        student_model: Student model in eval mode
        dataset: Dataset with image tensors and labels
        device: Device to run inference on

    Returns:
        DataFrame with predictions and metadata.

    """
    results = []

    for idx in range(len(dataset)):
        image_tensor, true_label = dataset[idx]

        teacher_pred, teacher_prob = get_prediction(teacher_model, image_tensor, device)
        student_pred, student_prob = get_prediction(student_model, image_tensor, device)

        result = {
            'idx': idx,
            'true_label': true_label,
            'teacher_pred': teacher_pred,
            'teacher_prob': teacher_prob,
            'student_pred': student_pred,
            'student_prob': student_prob,
            'teacher_correct': teacher_pred == true_label,
            'student_correct': student_pred == true_label,
        }

        # Add metadata if available
        if hasattr(dataset, 'df'):
            result['image_path'] = dataset.df.iloc[idx]['image_path']
            if 'lesion_type' in dataset.df.columns:
                result['lesion_type'] = dataset.df.iloc[idx]['lesion_type']
            if 'dx' in dataset.df.columns:
                result['dx'] = dataset.df.iloc[idx]['dx']

        results.append(result)

    return pd.DataFrame(results)


def load_original_image(image_path: str | pathlib.Path) -> Image.Image:
    """Load original image without transforms.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image in RGB format.

    """
    return Image.open(image_path).convert("RGB")


def plot_prediction(
    image: Image.Image,
    true_label: int,
    teacher_pred: int,
    teacher_prob: float,
    student_pred: int,
    student_prob: float,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot image with prediction results from both models.

    Args:
        image: PIL Image to display
        true_label: Ground truth label (0 or 1)
        teacher_pred: Teacher prediction
        teacher_prob: Teacher probability
        student_pred: Student prediction
        student_prob: Student probability
        title: Optional title for the plot
        ax: Matplotlib axes to plot on

    Returns:
        Matplotlib axes.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image)
    ax.axis('off')

    true_label_str = "MELANOMA" if true_label == 1 else "BENIGN"
    true_color = "red" if true_label == 1 else "green"

    teacher_correct = "+" if teacher_pred == true_label else "x"
    student_correct = "+" if student_pred == true_label else "x"

    ax.set_title(f"{title}\n" if title else "")

    text = f"Ground Truth: {true_label_str}\n"
    text += f"Teacher: {teacher_prob:.1%} {teacher_correct}\n"
    text += f"Student: {student_prob:.1%} {student_correct}"

    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for spine in ax.spines.values():
        spine.set_edgecolor(true_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

    return ax


def plot_prediction_grid(
    predictions_df: pd.DataFrame,
    filter_condition: str,
    title: str,
    n_samples: int = 8,
    save_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Plot a grid of predictions matching a filter condition.

    Args:
        predictions_df: DataFrame with prediction results
        filter_condition: String describing the filter (for title)
        title: Title for the figure
        n_samples: Maximum number of samples to show
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure.

    """
    samples = predictions_df.head(n_samples)
    n_actual = min(n_samples, len(samples))

    if n_actual == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No samples matching: {filter_condition}",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    n_cols = 4
    n_rows = (n_actual + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows + 1))
    axes = np.array(axes).flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= len(axes):
            break
        image = load_original_image(row['image_path'])
        lesion_type = row.get('lesion_type', '')[:25] if 'lesion_type' in row else ''
        plot_prediction(
            image, row['true_label'],
            row['teacher_pred'], row['teacher_prob'],
            row['student_pred'], row['student_prob'],
            title=lesion_type,
            ax=axes[i]
        )

    # Hide empty subplots
    for j in range(n_actual, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confidence_distribution(
    predictions_df: pd.DataFrame,
    save_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Plot confidence distributions for correct vs incorrect predictions.

    Args:
        predictions_df: DataFrame with prediction results
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure.

    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Teacher confidence distribution
    ax = axes[0]
    teacher_correct_probs = predictions_df[predictions_df['teacher_correct']]['teacher_prob']
    teacher_wrong_probs = predictions_df[~predictions_df['teacher_correct']]['teacher_prob']

    ax.hist(teacher_correct_probs, bins=30, alpha=0.6, label='Correct', color='green', density=True)
    ax.hist(teacher_wrong_probs, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', label='Decision boundary')
    ax.set_xlabel('Predicted Probability (Melanoma)')
    ax.set_ylabel('Density')
    ax.set_title('Teacher Confidence Distribution')
    ax.legend()

    # Student confidence distribution
    ax = axes[1]
    student_correct_probs = predictions_df[predictions_df['student_correct']]['student_prob']
    student_wrong_probs = predictions_df[~predictions_df['student_correct']]['student_prob']

    ax.hist(student_correct_probs, bins=30, alpha=0.6, label='Correct', color='green', density=True)
    ax.hist(student_wrong_probs, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', label='Decision boundary')
    ax.set_xlabel('Predicted Probability (Melanoma)')
    ax.set_ylabel('Density')
    ax.set_title('Student Confidence Distribution')
    ax.legend()

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def analyze_predictions(predictions_df: pd.DataFrame) -> dict[str, float | int]:
    """Analyze prediction results and return summary statistics.

    Args:
        predictions_df: DataFrame with prediction results

    Returns:
        Dictionary with analysis results.

    """
    teacher_acc = predictions_df['teacher_correct'].mean()
    student_acc = predictions_df['student_correct'].mean()

    teacher_correct_mask = predictions_df['teacher_correct']
    student_correct_mask = predictions_df['student_correct']

    both_correct = (teacher_correct_mask & student_correct_mask).sum()
    both_wrong = (~teacher_correct_mask & ~student_correct_mask).sum()
    teacher_only = (teacher_correct_mask & ~student_correct_mask).sum()
    student_only = (~teacher_correct_mask & student_correct_mask).sum()

    return {
        'teacher_accuracy': teacher_acc,
        'student_accuracy': student_acc,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'teacher_only_correct': teacher_only,
        'student_only_correct': student_only,
        'total_samples': len(predictions_df),
    }
