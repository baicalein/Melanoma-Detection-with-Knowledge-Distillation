"""Utility functions used across src."""

import logging
import pathlib
from typing import Any

# UVA color palette for consistent visualization
uva_colors = {
    "orange": "#E57200",
    "blue": "#232D4B",
    "light_blue": "#6CACE4",
    "gray": "#C1C6C8",
    "dark_gray": "#747678",
}


def setup_logger(
    name: str,
    log_file: pathlib.Path = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = pathlib.Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def count_parameters(model) -> dict[str, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_metrics(metrics: dict[str, Any], precision: int = 4) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.{precision}f}")
        else:
            lines.append(f"{key}: {value}")
    return " | ".join(lines)
