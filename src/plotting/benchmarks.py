"""Benchmark visualization utilities for model comparison.

Provides functions for:
- Loading benchmark results from JSON files
- Visualizing model comparisons
- Threshold tuning curves
- Knowledge distillation effectiveness plots
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def load_sklearn_results(artifacts_dir: pathlib.Path) -> pd.DataFrame:
    """Load sklearn baseline results from artifact summaries.
    
    Args:
        artifacts_dir: Path to the artifacts directory containing model results
        
    Returns:
        DataFrame with sklearn model results

    """
    results = []

    model_paths = {
        "Random Forest": "rf/rf_artifacts_summary.json",
        "Gradient Boosting": "gbm/gbm_artifacts_summary.json",
        "Logistic Regression": "logistic_regression/artifacts_summary.json",
    }

    for model_name, path in model_paths.items():
        full_path = artifacts_dir / path
        if full_path.exists():
            data = json.loads(full_path.read_text())
            for variant in ["baseline", "tuned"]:
                if variant in data:
                    results.append({
                        "Model": f"{model_name} ({variant})",
                        "Type": "sklearn",
                        "Accuracy": data[variant]["val"]["accuracy"],
                        "ROC-AUC": data[variant]["val"]["roc_auc"],
                    })

    return pd.DataFrame(results)


def load_teacher_checkpoints(checkpoints_dir: pathlib.Path) -> pd.DataFrame:
    """Load teacher model results from checkpoint metadata.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        DataFrame with teacher model results

    """
    results = []

    for meta_file in checkpoints_dir.glob("teacher_*_meta.json"):
        try:
            data = json.loads(meta_file.read_text())
            arch = meta_file.stem.replace("teacher_", "").replace("_focal_best_meta", "").replace("_best_meta", "")

            metrics = data.get("metrics", {})
            results.append({
                "Model": arch,
                "Type": "Teacher (DL)",
                "Epoch": data.get("epoch", "N/A"),
                "Accuracy": metrics.get("accuracy", np.nan),
                "Precision": metrics.get("precision", np.nan),
                "Recall": metrics.get("recall", np.nan),
                "F1": metrics.get("f1", np.nan),
                "Specificity": metrics.get("specificity", np.nan),
                "ROC-AUC": metrics.get("roc_auc", np.nan),
                "PR-AUC": metrics.get("pr_auc", np.nan),
                "ECE": metrics.get("ece", np.nan),
            })
        except Exception as e:
            print(f"Warning: Could not load {meta_file}: {e}")

    return pd.DataFrame(results)


def load_student_checkpoints(checkpoints_dir: pathlib.Path) -> pd.DataFrame:
    """Load student model results from checkpoint metadata.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        DataFrame with student model results

    """
    results = []

    for meta_file in checkpoints_dir.glob("student_*_meta.json"):
        try:
            data = json.loads(meta_file.read_text())
            config = meta_file.stem.replace("_best_meta", "").replace("student_", "")

            metrics = data.get("metrics", {})
            results.append({
                "Model": f"MobileNetV3 ({config})",
                "Type": "Student (KD)",
                "Epoch": data.get("epoch", "N/A"),
                "Accuracy": metrics.get("accuracy", np.nan),
                "F1": metrics.get("f1", np.nan),
                "ROC-AUC": metrics.get("roc_auc", np.nan),
                "PR-AUC": metrics.get("pr_auc", np.nan),
                "ECE": metrics.get("ece", np.nan),
            })
        except Exception as e:
            print(f"Warning: Could not load {meta_file}: {e}")

    return pd.DataFrame(results)


def plot_teacher_comparison(
    df_teachers: pd.DataFrame,
    save_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Visualize teacher model comparison.
    
    Args:
        df_teachers: DataFrame with teacher model results
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure

    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_viz = df_teachers.copy()
    df_viz["Family"] = df_viz["Model"].apply(
        lambda x: "ResNet" if "resnet" in x.lower() else "EfficientNet"
    )

    # Plot 1: ROC-AUC by model
    ax1 = axes[0]
    colors = ["#3498db" if "resnet" in m.lower() else "#e74c3c" for m in df_viz["Model"]]
    bars = ax1.barh(df_viz["Model"], df_viz["ROC-AUC"], color=colors, alpha=0.8)
    ax1.set_xlabel("ROC-AUC", fontsize=12)
    ax1.set_title("Teacher Model Performance (Validation ROC-AUC)", fontsize=14)
    ax1.set_xlim(0.80, 0.95)

    for bar, val in zip(bars, df_viz["ROC-AUC"]):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va='center', fontsize=9)

    legend_elements = [Patch(facecolor='#3498db', label='ResNet'),
                      Patch(facecolor='#e74c3c', label='EfficientNet')]
    ax1.legend(handles=legend_elements, loc='lower right')

    # Plot 2: Family comparison boxplot
    ax2 = axes[1]
    df_viz.boxplot(column="ROC-AUC", by="Family", ax=ax2)
    ax2.set_title("ROC-AUC by Architecture Family", fontsize=14)
    ax2.set_xlabel("Model Family", fontsize=12)
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    plt.suptitle("")

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_complete_model_comparison(
    df_all: pd.DataFrame,
    save_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Create complete model comparison visualization.
    
    Args:
        df_all: DataFrame with all model results
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure

    """
    df_sorted = df_all.sort_values("ROC-AUC", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.4)))

    color_map = {
        "sklearn": "#95a5a6",
        "Teacher (DL)": "#3498db",
        "Student (KD)": "#2ecc71",
    }
    colors = [color_map.get(t, "#bdc3c7") for t in df_sorted["Type"]]

    bars = ax.barh(range(len(df_sorted)), df_sorted["ROC-AUC"], color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["Model"], fontsize=10)
    ax.set_xlabel("ROC-AUC (Validation)", fontsize=12)
    ax.set_title("Complete Model Comparison - Melanoma Detection", fontsize=14, fontweight='bold')
    ax.set_xlim(0.75, 1.0)

    for i, (bar, val) in enumerate(zip(bars, df_sorted["ROC-AUC"])):
        ax.text(val + 0.005, i, f"{val:.4f}", va='center', fontsize=9)

    legend_elements = [
        Patch(facecolor='#95a5a6', label='sklearn Baselines'),
        Patch(facecolor='#3498db', label='Deep Learning Teachers'),
        Patch(facecolor='#2ecc71', label='Knowledge Distilled Student'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_kd_effectiveness(
    teacher_auc: float,
    student_auc: float,
    teacher_params: float = 25.0,
    student_params: float = 2.5,
    save_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Visualize knowledge distillation effectiveness.
    
    Args:
        teacher_auc: Teacher model ROC-AUC
        student_auc: Student model ROC-AUC
        teacher_params: Teacher parameters in millions
        student_params: Student parameters in millions
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    models = ["Best Teacher", "Best Student"]
    aucs = [teacher_auc, student_auc]
    params = [teacher_params, student_params]
    colors = ["#3498db", "#2ecc71"]

    # Performance comparison
    ax1 = axes[0]
    bars = ax1.bar(models, aucs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_ylabel("ROC-AUC", fontsize=12)
    ax1.set_title("Performance Comparison", fontsize=14)
    ax1.set_ylim(0.8, 1.0)
    for bar, val in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.4f}",
                ha='center', fontsize=12, fontweight='bold')

    # Size comparison
    ax2 = axes[1]
    bars = ax2.bar(models, params, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_ylabel("Parameters (Millions)", fontsize=12)
    ax2.set_title("Model Size Comparison", fontsize=14)
    for bar, val in zip(bars, params):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val}M",
                ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_threshold_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Plot threshold tuning curves for a single model.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        model_name: Name of the model for title
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure

    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    ax_roc = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC-AUC = {roc_auc_val:.4f}')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)

    idx_95 = np.argmin(np.abs(tpr - 0.95))
    ax_roc.scatter([fpr[idx_95]], [tpr[idx_95]], color='red', s=150, marker='*',
                   zorder=5, edgecolors='black', label=f'@95% sens: Spec={1-fpr[idx_95]:.3f}')
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
    ax_roc.set_title(f'{model_name} - ROC Curve', fontsize=12)
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)

    # PR Curve
    ax_pr = axes[1]
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(rec, prec)
    prevalence = y_true.mean()
    ax_pr.axhline(y=prevalence, color='gray', linestyle='--', label=f'Baseline (prevalence={prevalence:.3f})')
    ax_pr.plot(rec, prec, color='#3498db', lw=2, label=f'PR-AUC = {pr_auc_val:.4f}')

    idx_95_pr = np.argmin(np.abs(rec[:-1] - 0.95))
    ax_pr.scatter([rec[idx_95_pr]], [prec[idx_95_pr]], color='red', s=150, marker='*',
                  zorder=5, edgecolors='black', label=f'@95% recall: Prec={prec[idx_95_pr]:.3f}')
    ax_pr.set_xlabel('Recall (Sensitivity)')
    ax_pr.set_ylabel('Precision (PPV)')
    ax_pr.set_title(f'{model_name} - PR Curve', fontsize=12)
    ax_pr.legend(loc='lower left')
    ax_pr.grid(True, alpha=0.3)

    # Threshold curve
    ax_thresh = axes[2]
    thresholds = np.linspace(0.01, 0.99, 100)
    metrics = {'sensitivity': [], 'specificity': [], 'ppv': [], 'f1': []}

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0

        metrics['sensitivity'].append(sens)
        metrics['specificity'].append(spec)
        metrics['ppv'].append(ppv)
        metrics['f1'].append(f1)

    ax_thresh.plot(thresholds, metrics['sensitivity'], color='green', lw=2, label='Sensitivity')
    ax_thresh.plot(thresholds, metrics['specificity'], color='blue', lw=2, label='Specificity')
    ax_thresh.plot(thresholds, metrics['ppv'], color='orange', lw=2, label='PPV (Precision)')
    ax_thresh.plot(thresholds, metrics['f1'], color='purple', lw=2, linestyle='--', label='F1')
    ax_thresh.axhline(y=0.95, color='red', linestyle=':', alpha=0.7, label='95% target')

    sens_arr = np.array(metrics['sensitivity'])
    valid_idx = np.where(sens_arr >= 0.95)[0]
    if len(valid_idx) > 0:
        opt_thresh = thresholds[valid_idx[-1]]
        ax_thresh.axvline(x=opt_thresh, color='red', linestyle='-.', alpha=0.7,
                         label=f'Thresh @95% sens: {opt_thresh:.3f}')

    ax_thresh.set_xlabel('Classification Threshold')
    ax_thresh.set_ylabel('Metric Value')
    ax_thresh.set_title(f'{model_name} - Threshold vs Metrics', fontsize=12)
    ax_thresh.legend(loc='center right', fontsize=8)
    ax_thresh.grid(True, alpha=0.3)
    ax_thresh.set_xlim([0, 1])
    ax_thresh.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_latency_benchmarks(
    df_latency: pd.DataFrame,
    save_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Visualize latency benchmark results.
    
    Args:
        df_latency: DataFrame with latency benchmark results
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure

    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: CPU Latency comparison
    ax1 = axes[0]
    colors = ['#2ecc71' if 'Student' in m else '#3498db' for m in df_latency['Model']]
    bars = ax1.barh(df_latency['Model'], df_latency['CPU Latency (ms)'], color=colors, alpha=0.8)
    ax1.set_xlabel('Latency (ms)', fontsize=12)
    ax1.set_title('CPU Inference Latency\n(Lower is Better)', fontsize=14)
    for bar, val in zip(bars, df_latency['CPU Latency (ms)']):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}ms', va='center', fontsize=9)

    # Plot 2: Model Size vs Latency
    ax2 = axes[1]
    colors = ['#2ecc71' if 'Student' in m else '#3498db' for m in df_latency['Model']]
    ax2.scatter(df_latency['Size (MB)'], df_latency['CPU Latency (ms)'],
                c=colors, s=100, alpha=0.7, edgecolors='white', linewidth=2)
    for i, row in df_latency.iterrows():
        ax2.annotate(row['Model'].split()[0], (row['Size (MB)'], row['CPU Latency (ms)']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax2.set_xlabel('Model Size (MB)', fontsize=12)
    ax2.set_ylabel('CPU Latency (ms)', fontsize=12)
    ax2.set_title('Size vs Latency Tradeoff', fontsize=14)

    # Plot 3: Throughput comparison
    ax3 = axes[2]
    colors = ['#2ecc71' if 'Student' in m else '#3498db' for m in df_latency['Model']]
    bars = ax3.barh(df_latency['Model'], df_latency['CPU Throughput (FPS)'], color=colors, alpha=0.8)
    ax3.set_xlabel('Throughput (FPS)', fontsize=12)
    ax3.set_title('CPU Throughput\n(Higher is Better)', fontsize=14)
    for bar, val in zip(bars, df_latency['CPU Throughput (FPS)']):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=9)

    legend_elements = [Patch(facecolor='#3498db', label='Teacher'),
                      Patch(facecolor='#2ecc71', label='Student')]
    ax3.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_holdout_evaluation(
    df_holdout: pd.DataFrame,
    save_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Visualize holdout set evaluation results.
    
    Args:
        df_holdout: DataFrame with holdout evaluation results
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure

    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    width = 0.35

    # Plot 1: Performance metrics comparison
    ax1 = axes[0]
    metrics_to_plot = ['ROC-AUC', 'PR-AUC', 'F1', 'Sensitivity', 'Specificity']
    x = np.arange(len(metrics_to_plot))

    teacher_vals = [df_holdout.iloc[0][m] for m in metrics_to_plot]
    student_vals = [df_holdout.iloc[1][m] for m in metrics_to_plot]

    bars1 = ax1.bar(x - width/2, teacher_vals, width, label='Teacher', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, student_vals, width, label='Student', color='#2ecc71', alpha=0.8)

    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics (Holdout Set)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, teacher_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, student_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)

    # Plot 2: Clinical metrics at 95% sensitivity
    ax2 = axes[1]
    clinical_metrics = ['Spec @95% Sens', 'PPV @95% Sens']
    x2 = np.arange(len(clinical_metrics))

    teacher_clinical = [df_holdout.iloc[0][m] for m in clinical_metrics]
    student_clinical = [df_holdout.iloc[1][m] for m in clinical_metrics]

    bars1 = ax2.bar(x2 - width/2, teacher_clinical, width, label='Teacher', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x2 + width/2, student_clinical, width, label='Student', color='#2ecc71', alpha=0.8)

    ax2.set_ylabel('Score')
    ax2.set_title('Clinical Metrics @ 95% Sensitivity')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(clinical_metrics)
    ax2.legend()
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, teacher_clinical):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, student_clinical):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Deployment comparison
    ax3 = axes[2]
    deployment_metrics = ['Size (MB)', 'Latency (ms)']
    x3 = np.arange(len(deployment_metrics))

    teacher_deploy = [df_holdout.iloc[0][m] for m in deployment_metrics]
    student_deploy = [df_holdout.iloc[1][m] for m in deployment_metrics]

    bars1 = ax3.bar(x3 - width/2, teacher_deploy, width, label='Teacher', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x3 + width/2, student_deploy, width, label='Student', color='#2ecc71', alpha=0.8)

    ax3.set_ylabel('Value')
    ax3.set_title('Deployment Metrics (Lower is Better)')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(deployment_metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, teacher_deploy):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, student_deploy):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
