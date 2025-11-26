"""
Post-training quantization for mobile deployment.

Provides:
- Dynamic INT8 quantization
- Static INT8 quantization with calibration
- Model size and latency comparison
"""

import copy
import logging
import os
import pathlib
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.quantization import get_default_qconfig, quantize_dynamic

from src.config import CHECKPOINTS_DIR, QuantizationConfig
from src.evaluation.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """
    Apply dynamic quantization to model.
    
    Dynamic quantization quantizes weights but computes activations in float.
    Good for models with LSTM/Transformer layers or when calibration data unavailable.
    
    Args:
        model: PyTorch model to quantize
        dtype: Quantization dtype (qint8 or float16)
    
    Returns:
        Quantized model
    """
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    
    # Quantize Linear and Conv layers
    quantized_model = quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.Conv2d},
        dtype=dtype,
    )
    
    logger.info("Applied dynamic INT8 quantization")
    return quantized_model


def quantize_model_static(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    backend: str = "qnnpack",
    num_calibration_batches: int = 100,
) -> nn.Module:
    """
    Apply static quantization with calibration.
    
    Static quantization quantizes both weights and activations.
    Requires calibration data to determine activation ranges.
    
    Args:
        model: PyTorch model to quantize
        calibration_loader: DataLoader for calibration
        backend: Quantization backend ("qnnpack" for mobile, "fbgemm" for server)
        num_calibration_batches: Number of batches for calibration
    
    Returns:
        Quantized model
    """
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    
    # Set quantization backend
    torch.backends.quantized.engine = backend
    
    # Fuse modules (conv-bn-relu)
    # Note: This is model-specific and may need adjustment
    try:
        model_fused = torch.quantization.fuse_modules(
            model_cpu,
            [["conv", "bn", "relu"]],  # Adjust based on model structure
            inplace=False,
        )
    except Exception:
        model_fused = model_cpu
        logger.warning("Module fusion failed, proceeding without fusion")
    
    # Set qconfig
    model_fused.qconfig = get_default_qconfig(backend)
    
    # Prepare for quantization
    model_prepared = torch.quantization.prepare(model_fused, inplace=False)
    
    # Calibration
    logger.info(f"Calibrating with {num_calibration_batches} batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            images = images.cpu()
            model_prepared(images)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model_prepared, inplace=False)
    
    logger.info(f"Applied static INT8 quantization with {backend} backend")
    return quantized_model


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 ** 2)
        os.unlink(f.name)
    
    return size_mb


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Measure inference latency.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        device: Device for inference
        warmup: Warmup iterations
        iterations: Benchmark iterations
    
    Returns:
        Dict with latency statistics
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def compare_quantized_model(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Compare FP32 and quantized models.
    
    Args:
        fp32_model: Original FP32 model
        quantized_model: Quantized INT8 model
        dataloader: Validation dataloader
        device: Device for evaluation
    
    Returns:
        Comparison metrics including size, latency, and accuracy differences
    """
    # Model sizes
    fp32_size = get_model_size_mb(fp32_model)
    quant_size = get_model_size_mb(quantized_model)
    
    # Latency (quantized models run on CPU)
    fp32_latency = measure_latency(fp32_model, device="cpu")
    quant_latency = measure_latency(quantized_model, device="cpu")
    
    # Accuracy comparison
    def evaluate(model, loader, dev):
        model = model.to(dev)
        model.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(dev)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(targets.numpy())
        
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        return compute_classification_metrics(y_true, y_prob)
    
    fp32_metrics = evaluate(fp32_model, dataloader, device)
    quant_metrics = evaluate(quantized_model, dataloader, "cpu")
    
    return {
        "fp32": {
            "size_mb": fp32_size,
            "latency_ms": fp32_latency["mean_ms"],
            "roc_auc": fp32_metrics.roc_auc,
            "pr_auc": fp32_metrics.pr_auc,
            "ece": fp32_metrics.ece,
        },
        "int8": {
            "size_mb": quant_size,
            "latency_ms": quant_latency["mean_ms"],
            "roc_auc": quant_metrics.roc_auc,
            "pr_auc": quant_metrics.pr_auc,
            "ece": quant_metrics.ece,
        },
        "delta": {
            "size_reduction": 1 - (quant_size / fp32_size),
            "latency_speedup": fp32_latency["mean_ms"] / quant_latency["mean_ms"],
            "delta_roc_auc": quant_metrics.roc_auc - fp32_metrics.roc_auc,
            "delta_ece": quant_metrics.ece - fp32_metrics.ece,
        },
    }


def save_quantized_model(
    model: nn.Module,
    name: str,
    output_dir: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Save quantized model to disk."""
    output_dir = output_dir or CHECKPOINTS_DIR
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = output_dir / f"{name}_quantized.pth"
    torch.save(model.state_dict(), path)
    
    size_mb = get_model_size_mb(model)
    logger.info(f"Saved quantized model to {path} ({size_mb:.2f} MB)")
    
    return path
