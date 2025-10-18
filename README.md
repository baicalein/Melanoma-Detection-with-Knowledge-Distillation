# Deep Learning Final Project

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive, well-structured data science project template for academic research, powered by [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

> [!NOTE]
> This project template uses **PyTorch Lightning** for scalable training and **Weights & Biases** for experiment tracking, following modern deep learning best practices.

## 🎓 Project Overview

This project provides a cookie-cutter structure for deep learning research projects, specifically designed for academic use. It follows best practices for reproducibility, maintainability, and collaboration.

## 📁 Project Structure

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

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Deep_Learning_Final_Project
   ```

2. **Install uv** (if not already installed)

   ```bash
   pip install uv
   ```

3. **Create a virtual environment and install dependencies**

   ```bash
   # Install production dependencies
   uv sync
   
   # Install with development dependencies
   uv sync --extra dev
   
   # Install with all optional dependencies
   uv sync --all-extras
   ```

4. **Activate the virtual environment**

   ```bash
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

### Quick Start for Development

```bash
# Install all development tools
uv sync --extra dev

# Set up pre-commit hooks (recommended for code quality)
pre-commit install

# Start Jupyter Lab for notebooks
jupyter lab

# Run tests
pytest

# Format code with ruff (preferred) or black
ruff format src/ tests/
# or
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
```

## ⚡ Installation (1 minute)

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

## 🚀 Quick Commands

```bash
# Run tests
make test

# Format code
make format

# Start Jupyter Lab
make run-notebook

# See all available commands
make help
```

### Academic Writing

- [Scientific Python Lectures](https://lectures.scientific-python.org/)
- [Reproducible Research Best Practices](https://www.nature.com/articles/s41562-016-0021)

### Tools

- [uv Documentation](https://github.com/astral-sh/uv)
- [Pre-commit Hooks](https://pre-commit.com/)

## 📧 Contact

For questions or collaboration opportunities, please contact [your contact information].

---

**Note**: This is a template project. Customize it according to your specific research needs.
