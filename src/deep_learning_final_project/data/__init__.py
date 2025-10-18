"""
Data loading and preprocessing modules.

This package contains utilities for loading, preprocessing, and augmenting data
for deep learning research projects.
"""

from pathlib import Path

# Version information
__version__ = "0.1.0"

# Data directories - can be configured via environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"

__all__ = [
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "INTERIM_DATA_DIR",
]
