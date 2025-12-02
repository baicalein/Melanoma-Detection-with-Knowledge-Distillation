"""Evaluation metrics, inference, and quantization utilities."""

from src.evaluation.inference import (
    analyze_predictions,
    get_all_predictions,
    get_model_predictions,
    get_prediction,
    load_model_from_checkpoint,
    load_original_image,
    plot_confidence_distribution,
    plot_prediction,
    plot_prediction_grid,
)
from src.evaluation.metrics import (
    ClassificationMetrics,
    DeploymentMetrics,
    compute_calibration_error,
    compute_classification_metrics,
    compute_deployment_metrics,
    evaluate_model,
)
from src.evaluation.quantization import (
    compare_quantized_model,
    quantize_model_dynamic,
    quantize_model_static,
)

__all__ = [
    "ClassificationMetrics",
    "DeploymentMetrics",
    "compute_classification_metrics",
    "evaluate_model",
    "compute_calibration_error",
    "compute_deployment_metrics",
    "quantize_model_dynamic",
    "quantize_model_static",
    "compare_quantized_model",
    "load_model_from_checkpoint",
    "get_prediction",
    "get_all_predictions",
    "get_model_predictions",
    "load_original_image",
    "plot_prediction",
    "plot_prediction_grid",
    "plot_confidence_distribution",
    "analyze_predictions",
]
