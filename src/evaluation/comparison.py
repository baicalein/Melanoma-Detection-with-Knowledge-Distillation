"""Model comparison utilities for Teacher-Student analysis.

Provides functions for:
- Comparing predictions between Teacher and Student models
- Analyzing disagreements
- Generating comparison visualizations
- Computing comparison metrics
"""

import pathlib
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.config import RAW_DIR


@dataclass
class ComparisonMetrics:
    """Container for model comparison metrics."""

    roc_auc: float
    pr_auc: float
    f1: float
    accuracy: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "ROC-AUC": self.roc_auc,
            "PR-AUC": self.pr_auc,
            "F1": self.f1,
            "Accuracy": self.accuracy,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> ComparisonMetrics:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        y_pred: Predicted labels

    Returns:
        ComparisonMetrics with all computed metrics.
    """
    return ComparisonMetrics(
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred),
        accuracy=(y_pred == y_true).mean(),
    )


@torch.no_grad()
def get_batch_predictions(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Get predictions from a model on the entire dataset using batching.

    Args:
        model: PyTorch model in eval mode
        dataset: Dataset to run inference on
        device: Device for inference
        batch_size: Batch size for inference

    Returns:
        Dictionary with logits, probs, labels, and preds.
    """
    model.eval()

    all_logits = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    for batch in tqdm(dataloader, desc="Running inference"):
        images, labels = batch
        images = images.to(device)
        logits = model(images)

        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits).squeeze()
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)

    return {
        "logits": logits.numpy(),
        "probs": probs.numpy(),
        "labels": labels.numpy(),
        "preds": (probs > 0.5).numpy().astype(int),
    }


def build_comparison_dataframe(
    holdout_df: pd.DataFrame,
    teacher_results: dict[str, np.ndarray],
    student_results: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build a comparison DataFrame with predictions from both models.

    Args:
        holdout_df: Original holdout DataFrame with image_id and image_path
        teacher_results: Results from get_batch_predictions for teacher
        student_results: Results from get_batch_predictions for student

    Returns:
        DataFrame with all comparison columns.
    """
    comparison_df = pd.DataFrame(
        {
            "image_id": holdout_df["image_id"].values,
            "image_path": holdout_df["image_path"].values,
            "true_label": teacher_results["labels"],
            "teacher_prob": teacher_results["probs"],
            "teacher_pred": teacher_results["preds"],
            "student_prob": student_results["probs"],
            "student_pred": student_results["preds"],
        }
    )

    # Add derived columns
    comparison_df["prob_diff"] = (
        comparison_df["teacher_prob"] - comparison_df["student_prob"]
    )
    comparison_df["abs_prob_diff"] = comparison_df["prob_diff"].abs()
    comparison_df["models_agree"] = (
        comparison_df["teacher_pred"] == comparison_df["student_pred"]
    )
    comparison_df["teacher_correct"] = (
        comparison_df["teacher_pred"] == comparison_df["true_label"]
    )
    comparison_df["student_correct"] = (
        comparison_df["student_pred"] == comparison_df["true_label"]
    )

    return comparison_df


def analyze_disagreements(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze disagreement cases between models.

    Args:
        comparison_df: DataFrame from build_comparison_dataframe

    Returns:
        DataFrame containing only disagreement cases with analysis columns.
    """
    disagreements = comparison_df[~comparison_df["models_agree"]].copy()

    # Categorize disagreements
    disagreements["category"] = "Unknown"
    disagreements.loc[
        (disagreements["teacher_pred"] == 1) & (disagreements["student_pred"] == 0),
        "category",
    ] = "Teacher predicts Melanoma, Student predicts Benign"
    disagreements.loc[
        (disagreements["teacher_pred"] == 0) & (disagreements["student_pred"] == 1),
        "category",
    ] = "Teacher predicts Benign, Student predicts Melanoma"

    # Add correctness analysis
    disagreements["who_is_right"] = "Neither"
    disagreements.loc[
        disagreements["teacher_correct"] & ~disagreements["student_correct"],
        "who_is_right",
    ] = "Teacher"
    disagreements.loc[
        ~disagreements["teacher_correct"] & disagreements["student_correct"],
        "who_is_right",
    ] = "Student"
    disagreements.loc[
        disagreements["teacher_correct"] & disagreements["student_correct"],
        "who_is_right",
    ] = "Both"

    return disagreements


def resolve_image_path(image_path: str, raw_dir: pathlib.Path = RAW_DIR) -> pathlib.Path:
    """Resolve image path relative to RAW_DIR if needed.

    Args:
        image_path: Path string from DataFrame
        raw_dir: Base directory for raw images

    Returns:
        Resolved absolute path.
    """
    img_path = pathlib.Path(image_path)
    if img_path.is_absolute() and img_path.exists():
        return img_path
    # Try relative to RAW_DIR
    candidate = raw_dir / image_path
    if candidate.exists():
        return candidate
    return img_path


def plot_sample_images(
    df: pd.DataFrame,
    title: str,
    n_samples: int = 8,
    figsize: tuple[int, int] = (16, 8),
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """Plot sample images with predictions.

    Args:
        df: DataFrame with image_path, true_label, teacher_prob, student_prob
        title: Title for the figure
        n_samples: Maximum number of samples to show
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure.
    """
    samples = df.sample(n=min(n_samples, len(df)), random_state=42)

    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for ax, (_, row) in zip(axes, samples.iterrows()):
        # Load image
        try:
            img_path = resolve_image_path(row["image_path"])
            img = Image.open(img_path)
            ax.imshow(img)
        except (FileNotFoundError, OSError):
            ax.text(
                0.5,
                0.5,
                "Image not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # Create title with predictions
        true_label = "MELANOMA" if row["true_label"] == 1 else "BENIGN"
        teacher_pred = f"Teacher: {row['teacher_prob']:.2f}"
        student_pred = f"Student: {row['student_prob']:.2f}"

        # Color based on ground truth
        title_color = "#e74c3c" if row["true_label"] == 1 else "#2ecc71"

        ax.set_title(
            f"True: {true_label}\n{teacher_pred} | {student_pred}",
            fontsize=10,
            color=title_color,
            fontweight="bold",
        )
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_probability_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """Plot probability distribution comparison between models.

    Args:
        comparison_df: DataFrame from build_comparison_dataframe
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Teacher vs Student probabilities
    ax1 = axes[0]
    colors = [
        "#2ecc71" if l == 0 else "#e74c3c" for l in comparison_df["true_label"]
    ]
    ax1.scatter(
        comparison_df["teacher_prob"],
        comparison_df["student_prob"],
        c=colors,
        alpha=0.5,
        s=20,
    )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect agreement")
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Teacher Probability", fontsize=12)
    ax1.set_ylabel("Student Probability", fontsize=12)
    ax1.set_title("Teacher vs Student Predictions", fontsize=14)
    ax1.legend(["Diagonal", "Benign", "Melanoma"], loc="upper left")

    # Plot 2: Probability distributions
    ax2 = axes[1]
    ax2.hist(
        comparison_df["teacher_prob"],
        bins=50,
        alpha=0.5,
        label="Teacher",
        color="#3498db",
    )
    ax2.hist(
        comparison_df["student_prob"],
        bins=50,
        alpha=0.5,
        label="Student",
        color="#e74c3c",
    )
    ax2.axvline(0.5, color="black", linestyle="--", label="Threshold")
    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Probability Distributions", fontsize=14)
    ax2.legend()

    # Plot 3: Probability difference distribution
    ax3 = axes[2]
    ax3.hist(comparison_df["prob_diff"], bins=50, color="#9b59b6", alpha=0.7)
    ax3.axvline(0, color="black", linestyle="--")
    ax3.set_xlabel("Probability Difference (Teacher - Student)", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Prediction Differences", fontsize=14)

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_disagreement_analysis(
    disagreements: pd.DataFrame,
    save_path: Optional[pathlib.Path] = None,
) -> plt.Figure:
    """Plot disagreement analysis charts.

    Args:
        disagreements: DataFrame from analyze_disagreements
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Disagreement categories
    ax1 = axes[0]
    category_counts = disagreements["category"].value_counts()
    colors = ["#e74c3c", "#3498db"]
    ax1.barh(range(len(category_counts)), category_counts.values, color=colors)
    ax1.set_yticks(range(len(category_counts)))
    ax1.set_yticklabels(
        [c.replace(", ", ",\n") for c in category_counts.index], fontsize=10
    )
    ax1.set_xlabel("Count", fontsize=12)
    ax1.set_title("Types of Disagreements", fontsize=14)
    for i, v in enumerate(category_counts.values):
        ax1.text(v + 1, i, str(v), va="center", fontsize=11)

    # Plot 2: Who is correct?
    ax2 = axes[1]
    right_counts = disagreements["who_is_right"].value_counts()
    colors_right = {
        "Teacher": "#3498db",
        "Student": "#e74c3c",
        "Neither": "#95a5a6",
        "Both": "#2ecc71",
    }
    bars = ax2.bar(
        right_counts.index,
        right_counts.values,
        color=[colors_right.get(x, "#95a5a6") for x in right_counts.index],
    )
    ax2.set_xlabel("Who is Correct", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Correctness in Disagreements", fontsize=14)
    for bar, val in zip(bars, right_counts.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val),
            ha="center",
            fontsize=11,
        )

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_comparison_summary(
    comparison_df: pd.DataFrame,
    disagreements: pd.DataFrame,
) -> dict[str, any]:
    """Generate a summary of the model comparison.

    Args:
        comparison_df: DataFrame from build_comparison_dataframe
        disagreements: DataFrame from analyze_disagreements

    Returns:
        Dictionary with summary statistics.
    """
    both_wrong = comparison_df[
        (~comparison_df["teacher_correct"]) & (~comparison_df["student_correct"])
    ]

    return {
        "Total Samples": len(comparison_df),
        "Models Agree": comparison_df["models_agree"].sum(),
        "Models Disagree": (~comparison_df["models_agree"]).sum(),
        "Agreement Rate": comparison_df["models_agree"].mean(),
        "Teacher Correct": comparison_df["teacher_correct"].sum(),
        "Student Correct": comparison_df["student_correct"].sum(),
        "Both Correct": (
            comparison_df["teacher_correct"] & comparison_df["student_correct"]
        ).sum(),
        "Both Wrong": len(both_wrong),
        "Disagreements - Teacher Right": (
            (disagreements["who_is_right"] == "Teacher").sum()
            if len(disagreements) > 0
            else 0
        ),
        "Disagreements - Student Right": (
            (disagreements["who_is_right"] == "Student").sum()
            if len(disagreements) > 0
            else 0
        ),
    }


def predict_single_image(
    image_path: str,
    teacher_model: nn.Module,
    student_model: nn.Module,
    transform: callable,
    device: torch.device,
) -> dict[str, any]:
    """Run inference on a single image and compare models.

    Args:
        image_path: Path to the image
        teacher_model: Teacher model in eval mode
        student_model: Student model in eval mode
        transform: Preprocessing transform
        device: Device for inference

    Returns:
        Dictionary with image and prediction results.
    """
    # Resolve path and load image
    img_path = resolve_image_path(image_path)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        teacher_logit = teacher_model(img_tensor)
        student_logit = student_model(img_tensor)

    teacher_prob = torch.sigmoid(teacher_logit).item()
    student_prob = torch.sigmoid(student_logit).item()

    return {
        "image": img,
        "teacher_prob": teacher_prob,
        "student_prob": student_prob,
        "teacher_pred": "Melanoma" if teacher_prob > 0.5 else "Benign",
        "student_pred": "Melanoma" if student_prob > 0.5 else "Benign",
        "agreement": (teacher_prob > 0.5) == (student_prob > 0.5),
    }
