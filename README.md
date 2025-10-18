# Deep Learning Final Project

A comprehensive, well-structured data science project template for academic research, powered by [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

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

# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
```

## 📊 Data Management

### Best Practices for Academic Research

1. **Keep raw data immutable**: Never modify files in `data/raw/`
2. **Document data sources**: Create a `data/README.md` with data provenance
3. **Version control small datasets**: Use Git LFS for datasets < 100MB
4. **Use external storage for large datasets**: Store large files on institutional servers or cloud storage
5. **Create reproducible preprocessing**: All transformations should be in code, not manual

### Data Organization

- `data/raw/`: Store original datasets as downloaded (read-only)
- `data/interim/`: Intermediate processing steps
- `data/processed/`: Final datasets ready for modeling
- `data/external/`: Reference data, supplementary datasets

## 🧪 Experimentation Workflow

### Running Experiments

1. **Create an experiment configuration**
   ```bash
   # configs/experiments/experiment_001.yaml
   ```

2. **Run training**
   ```bash
   python -m deep_learning_final_project.scripts.train --config configs/experiments/experiment_001.yaml
   ```

3. **Monitor with TensorBoard**
   ```bash
   tensorboard --logdir logs/
   ```

4. **Evaluate results**
   ```bash
   python -m deep_learning_final_project.scripts.evaluate --checkpoint checkpoints/best_model.pth
   ```

### Tracking Experiments

For academic reproducibility, always:
- [ ] Log hyperparameters in configuration files
- [ ] Save random seeds for reproducibility
- [ ] Record environment details (Python version, package versions)
- [ ] Save model checkpoints at key milestones
- [ ] Document results in `results/metrics/`

## 📝 Notebook Organization

### Exploratory Notebooks (`notebooks/exploratory/`)
- Use for initial data exploration
- Name with dates: `YYYY-MM-DD-description.ipynb`
- Example: `2024-01-15-initial-eda.ipynb`

### Report Notebooks (`notebooks/reports/`)
- Polished notebooks for final results
- Use for generating figures and tables for papers
- Name descriptively: `model-comparison.ipynb`, `results-visualization.ipynb`

### Notebook Best Practices
```python
# Start each notebook with imports and setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/deep_learning_final_project

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

## 📖 Documentation

### Code Documentation
- Use docstrings for all functions and classes (Google or NumPy style)
- Add type hints for better code clarity
- Document complex algorithms with inline comments

### Research Documentation
- Keep a research journal in `docs/journal.md`
- Document design decisions in `docs/decisions.md`
- Maintain a bibliography in `docs/references.bib`

## 🔧 Development Tools

This project uses modern Python development tools:
- **uv**: Fast package management and virtual environments
- **ruff**: Fast Python linter (replaces flake8, pylint)
- **black**: Code formatter
- **isort**: Import sorter
- **pytest**: Testing framework
- **mypy**: Static type checker
- **pre-commit**: Git hooks for code quality

## 📦 Package Management with uv

```bash
# Add a new dependency
uv add numpy pandas

# Add a development dependency
uv add --dev pytest jupyter

# Remove a dependency
uv remove package-name

# Update all dependencies
uv lock --upgrade

# Show dependency tree
uv tree
```

## 🎯 Academic Research Tips

### Reproducibility Checklist
- [ ] Set random seeds everywhere (Python, NumPy, PyTorch)
- [ ] Pin exact package versions in `uv.lock`
- [ ] Document hardware specifications in `docs/environment.md`
- [ ] Save hyperparameters in version control
- [ ] Use configuration files instead of command-line arguments
- [ ] Log all experiments with timestamps and git commits

### Collaboration Tips
- Use clear commit messages describing intent
- Review code together during team meetings
- Document assumptions in code comments
- Use branches for experimental features
- Keep the main branch stable and tested

### Paper Writing Integration
1. Generate all figures in `results/figures/` using scripts
2. Export tables to LaTeX format in `results/tables/`
3. Keep a separate `paper/` directory for manuscript drafts
4. Link code to specific sections in your paper

## 🤝 Contributing

For academic collaborators:
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and add tests
3. Run code quality checks: `ruff check . && black . && pytest`
4. Commit your changes: `git commit -m "Add your feature"`
5. Push and create a pull request

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- List your institution, funding sources, and collaborators
- Cite any codebases or papers that inspired this work
- Acknowledge any data sources used

## 📚 Useful Resources

### Deep Learning Frameworks
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

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