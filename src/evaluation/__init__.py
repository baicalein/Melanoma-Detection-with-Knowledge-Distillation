"""Evaluation metrics and quantization utilities."""

from src.evaluation.metrics import (
    ClassificationMetrics,
    DeploymentMetrics,
    compute_classification_metrics,
    evaluate_model,
    compute_calibration_error,
    compute_deployment_metrics,
)
from src.evaluation.quantization import (
    quantize_model_dynamic,
    quantize_model_static,
    compare_quantized_model,
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
]
