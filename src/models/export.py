"""Model export utilities for deployment.

Provides functions for:
- Exporting models to ONNX format
- Exporting models to TorchScript format
- Validating exported models
- Benchmarking inference speed
"""

import pathlib
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ExportResult:
    """Container for export results."""

    path: pathlib.Path
    format: str
    size_mb: float
    validated: bool


def export_to_onnx(
    model: nn.Module,
    model_name: str,
    dummy_input: torch.Tensor,
    export_dir: pathlib.Path,
    opset_version: int = 14,
) -> ExportResult:
    """Export model to ONNX format.

    Args:
        model: PyTorch model to export
        model_name: Name for the exported file
        dummy_input: Sample input tensor for tracing
        export_dir: Directory to save exported model
        opset_version: ONNX opset version

    Returns:
        ExportResult with path and metadata.

    """
    model.eval()
    export_dir = pathlib.Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / f"{model_name}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False
    )

    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"Exported {model_name} to ONNX: {onnx_path.name} ({size_mb:.2f} MB)")

    return ExportResult(
        path=onnx_path,
        format="ONNX",
        size_mb=size_mb,
        validated=False
    )


def export_to_torchscript(
    model: nn.Module,
    model_name: str,
    dummy_input: torch.Tensor,
    export_dir: pathlib.Path,
    method: str = 'trace',
) -> ExportResult:
    """Export model to TorchScript format.

    Args:
        model: PyTorch model to export
        model_name: Name for the exported file
        dummy_input: Sample input tensor for tracing
        export_dir: Directory to save exported model
        method: 'trace' or 'script'

    Returns:
        ExportResult with path and metadata.

    """
    model.eval()
    export_dir = pathlib.Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    ts_path = export_dir / f"{model_name}.pt"

    with torch.no_grad():
        if method == 'trace':
            traced_model = torch.jit.trace(model, dummy_input)
        else:
            traced_model = torch.jit.script(model)

    traced_model = torch.jit.optimize_for_inference(traced_model)
    traced_model.save(str(ts_path))

    size_mb = ts_path.stat().st_size / (1024 ** 2)
    print(f"Exported {model_name} to TorchScript: {ts_path.name} ({size_mb:.2f} MB)")

    return ExportResult(
        path=ts_path,
        format="TorchScript",
        size_mb=size_mb,
        validated=False
    )


def validate_onnx(
    onnx_path: pathlib.Path,
    pytorch_model: nn.Module,
    dummy_input: torch.Tensor,
    tolerance: float = 1e-4,
) -> bool:
    """Validate ONNX model matches PyTorch output.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        dummy_input: Sample input tensor
        tolerance: Maximum allowed difference

    Returns:
        True if validation passes.

    """
    import onnx
    import onnxruntime as ort

    # Check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Run ONNX inference
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]

    # Run PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()

    diff = np.abs(onnx_output - pytorch_output).max()
    passed = diff < tolerance

    print(f"{onnx_path.name}: Max difference = {diff:.2e} ({'PASS' if passed else 'FAIL'})")
    return passed


def validate_torchscript(
    ts_path: pathlib.Path,
    pytorch_model: nn.Module,
    dummy_input: torch.Tensor,
    tolerance: float = 1e-4,
) -> bool:
    """Validate TorchScript model matches PyTorch output.

    Args:
        ts_path: Path to TorchScript model
        pytorch_model: Original PyTorch model
        dummy_input: Sample input tensor
        tolerance: Maximum allowed difference

    Returns:
        True if validation passes.

    """
    loaded_model = torch.jit.load(str(ts_path))
    loaded_model.eval()

    with torch.no_grad():
        ts_output = loaded_model(dummy_input).cpu().numpy()

    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()

    diff = np.abs(ts_output - pytorch_output).max()
    passed = diff < tolerance

    print(f"{ts_path.name}: Max difference = {diff:.2e} ({'PASS' if passed else 'FAIL'})")
    return passed


def benchmark_pytorch_inference(
    model: nn.Module,
    dummy_input: torch.Tensor,
    n_runs: int = 100,
    warmup: int = 10,
    name: str = "Model",
) -> tuple[float, float]:
    """Benchmark PyTorch model inference time.

    Args:
        model: Model to benchmark
        dummy_input: Sample input tensor
        n_runs: Number of inference runs
        warmup: Number of warmup runs
        name: Model name for printing

    Returns:
        Tuple of (mean_time_ms, std_time_ms).

    """
    model.eval()
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"{name}: {mean_time:.2f} +/- {std_time:.2f} ms")
    return mean_time, std_time


def benchmark_onnx_inference(
    onnx_path: pathlib.Path,
    dummy_input: torch.Tensor,
    n_runs: int = 100,
    warmup: int = 10,
    name: str = "Model",
) -> tuple[float, float]:
    """Benchmark ONNX model inference time.

    Args:
        onnx_path: Path to ONNX model
        dummy_input: Sample input tensor
        n_runs: Number of inference runs
        warmup: Number of warmup runs
        name: Model name for printing

    Returns:
        Tuple of (mean_time_ms, std_time_ms).

    """
    import onnxruntime as ort

    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}

    # Warmup
    for _ in range(warmup):
        _ = ort_session.run(None, onnx_input)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = ort_session.run(None, onnx_input)
        times.append((time.perf_counter() - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"{name}: {mean_time:.2f} +/- {std_time:.2f} ms")
    return mean_time, std_time


def get_export_summary(export_dir: pathlib.Path) -> list[dict[str, str]]:
    """Get summary of all exported models in directory.

    Args:
        export_dir: Directory containing exported models

    Returns:
        List of dicts with file info.

    """
    export_dir = pathlib.Path(export_dir)
    summary = []

    for f in sorted(export_dir.glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 ** 2)
            summary.append({
                'File': f.name,
                'Format': f.suffix[1:].upper(),
                'Size (MB)': f"{size_mb:.2f}"
            })

    return summary
