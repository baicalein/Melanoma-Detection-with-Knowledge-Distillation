# Deep Learning Final Project

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


TODO: fill this in

##  Project Structure

```
.
├── src/                          # Source code for the project
│   └── deep_learning_final_project/
│       ├── data/                 # Data loading and preprocessing modules
│       ├── models/               # Model architectures and definitions
│       ├── utils/                # Utility functions and helpers
│       ├── scripts/              # Training, evaluation, and inference scripts
│       └── visualization/        # Plotting and visualization utilities
├── data/                         # Data directory (add to .gitignore for large files)
│   ├── raw/                      # Original, immutable data
│   ├── processed/                # Cleaned and transformed data
│   ├── external/                 # External datasets or reference data
│   └── interim/                  # Intermediate data processing steps
├── notebooks/                    # Jupyter notebooks for exploration and reporting
│   ├── exploratory/              # Exploratory data analysis (EDA)
│   └── reports/                  # Final analysis and visualizations for papers
├── tests/                        # Unit tests and integration tests
├── configs/                      # Configuration files (YAML/JSON)
│   ├── experiments/              # Experiment-specific configs
│   └── models/                   # Model architecture configs
├── scripts/                      # Standalone scripts for specific tasks
├── docs/                         # Documentation and research notes
├── models/                       # Saved model architectures (definitions only)
├── checkpoints/                  # Model checkpoints (add to .gitignore)
├── logs/                         # Training logs, TensorBoard files
├── results/                      # Experiment results
│   ├── figures/                  # Generated figures for papers
│   ├── tables/                   # Result tables and metrics
│   └── metrics/                  # Numerical results and performance metrics
├── pyproject.toml               # Project dependencies and configuration
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager


## Installation 

```bash
# Clone the repository
git clone <your-repo-url>
cd Deep_Learning_Final_Project

# Install uv (if not already installed)
pip install uv

# Install dependencies
uv sync --extra dev

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

## Quick setup

```bash
  make install        - Install production dependencies
  make install-dev    - Install development dependencies
  make test          - Run tests with coverage
  make lint          - Run linters (ruff, mypy)
  make format        - Format code (black, isort)
  make clean         - Clean build artifacts and cache
  make run-notebook  - Start Jupyter Lab
  make train         - Run example training script
  make docs          - Build documentation
  make pre-commit    - Install pre-commit hooks
  make check-all     - Run format, lint, and test
```

### Academic Writing

- [Scientific Python Lectures](https://lectures.scientific-python.org/)
- [Reproducible Research Best Practices](https://www.nature.com/articles/s41562-016-0021)

### Tools

- [uv Documentation](https://github.com/astral-sh/uv)
- [Pre-commit Hooks](https://pre-commit.com/)

## Contact

Ryan Healy and Angie Yoon