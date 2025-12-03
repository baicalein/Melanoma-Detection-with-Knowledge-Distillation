"""Evaluation metrics, inference, and quantization utilities."""

from src.evaluation.comparison import (
    ComparisonMetrics,
    analyze_disagreements,
    build_comparison_dataframe,
    compute_metrics,
    generate_comparison_summary,
    get_batch_predictions,
    plot_disagreement_analysis,
    plot_probability_comparison,
    plot_sample_images,
    predict_single_image,
    resolve_image_path,
)
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
    # Comparison
    "ComparisonMetrics",
    "analyze_disagreements",
    "build_comparison_dataframe",
    "compute_metrics",
    "generate_comparison_summary",
    "get_batch_predictions",
    "plot_disagreement_analysis",
    "plot_probability_comparison",
    "plot_sample_images",
    "predict_single_image",
    "resolve_image_path",
    # Metrics
    "ClassificationMetrics",
    "DeploymentMetrics",
    "compute_classification_metrics",
    "evaluate_model",
    "compute_calibration_error",
    "compute_deployment_metrics",
    # Quantization
    "quantize_model_dynamic",
    "quantize_model_static",
    "compare_quantized_model",
    # Inference
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
