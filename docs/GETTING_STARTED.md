# Getting Started Guide

This guide will help you set up the project and run your first experiment.

> [!NOTE]
> This project uses **PyTorch Lightning** for training and **Weights & Biases** for experiment tracking. Make sure you have a W&B account (free for academics).

## 📋 Prerequisites

- Python 3.12 or higher
- Git
- CUDA-capable GPU (optional, but recommended for deep learning)
- [Weights & Biases account](https://wandb.ai/signup) (free for academic use)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repository-url>
cd Deep_Learning_Final_Project

# Install uv if not already installed
pip install uv

# Create virtual environment and install dependencies
uv sync --extra dev

# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

> [!TIP]
> Use `uv sync --all-extras` to install all optional dependencies including NLP and CV libraries.

### 2. Verify Installation

```bash
# Check that packages are installed
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'Lightning version: {pl.__version__}')"
python -c "import wandb; print(f'W&B version: {wandb.__version__}')"

# Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

> [!WARNING]
> If CUDA is not available but you have a GPU, you may need to reinstall PyTorch with CUDA support. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 3. Set Up Your Data

```bash
# Create data directories (already created by template)
# Place your raw data in data/raw/

# Example: Download a dataset
# wget https://example.com/dataset.zip -O data/raw/dataset.zip
# unzip data/raw/dataset.zip -d data/raw/
```

### 4. Configure Your Experiment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor

# Review and customize experiment config
nano configs/experiments/baseline.yaml
```

### 5. Run Your First Experiment

```bash
# Train a model
python -m deep_learning_final_project.scripts.train \
    --config configs/experiments/baseline.yaml

# Evaluate the model
python -m deep_learning_final_project.scripts.evaluate \
    --checkpoint checkpoints/best_model.pth
```

## 📊 Working with Notebooks

### Start Jupyter Lab

```bash
# Make sure dev dependencies are installed
uv sync --extra dev

# Start Jupyter Lab
jupyter lab
```

### Create Your First Notebook

1. Navigate to `notebooks/exploratory/`
2. Create a new notebook: `YYYY-MM-DD-initial-exploration.ipynb`
3. Start with this template:

```python
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
%matplotlib inline

# Import project modules
from deep_learning_final_project.data import DATA_DIR, RAW_DATA_DIR

# Your exploration code here
print(f"Data directory: {DATA_DIR}")
```

## 🔬 Typical Research Workflow

### 1. Data Exploration

```bash
# Create an exploratory notebook
cd notebooks/exploratory/
# Create: 2024-01-01-dataset-exploration.ipynb

# Explore your data:
# - Check data statistics
# - Visualize distributions
# - Identify data quality issues
# - Document findings
```

### 2. Data Preprocessing

```python
# In your notebook or preprocessing script
from deep_learning_final_project.data import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Load raw data
raw_data = load_your_data(RAW_DATA_DIR / "dataset.csv")

# Process data
processed_data = preprocess(raw_data)

# Save processed data
processed_data.to_csv(PROCESSED_DATA_DIR / "processed_dataset.csv")
```

### 3. Model Development

```python
# src/deep_learning_final_project/models/my_model.py
from deep_learning_final_project.models.base_model import BaseModel

class MyResearchModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Your model architecture
        
    def forward(self, x):
        # Your forward pass
        return x
```

### 4. Training

```bash
# Create experiment config
cp configs/experiments/baseline.yaml configs/experiments/my_experiment.yaml

# Edit configuration
nano configs/experiments/my_experiment.yaml

# Run training
python -m deep_learning_final_project.scripts.train \
    --config configs/experiments/my_experiment.yaml
```

### 5. Evaluation and Analysis

```bash
# Evaluate on test set
python -m deep_learning_final_project.scripts.evaluate \
    --checkpoint checkpoints/my_experiment/best.pth

# Create report notebook in notebooks/reports/
# Generate figures for your paper in results/figures/
```

## 🛠️ Development Tools

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/deep_learning_final_project

# Run specific test file
pytest tests/test_models.py -v
```

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# Open browser to http://localhost:6006
```

## 📝 Best Practices

### 1. Version Control

```bash
# Make frequent, small commits
git add src/my_module.py
git commit -m "Add feature X to module Y"

# Use descriptive branch names
git checkout -b feature/new-model-architecture
git checkout -b fix/data-loading-bug
```

### 2. Experiment Tracking

- Create a new config file for each experiment
- Name configs descriptively: `experiment_001_resnet50.yaml`
- Document hyperparameters and results in `docs/experiments.md`

### 3. Data Management

- Never modify files in `data/raw/`
- Document data sources in `data/README.md`
- Use version control for small datasets
- Store large datasets externally and document URLs

### 4. Reproducibility

- Set random seeds everywhere
- Pin package versions (automatic with uv)
- Document hardware specifications
- Save configuration with results

## 🆘 Troubleshooting

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should be True

# If False, reinstall PyTorch with CUDA:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Errors

- Reduce batch size in config
- Use gradient accumulation
- Enable mixed precision training
- Use smaller model or image size

### Import Errors

```bash
# Make sure you're in the virtual environment
which python  # Should point to .venv/bin/python

# Reinstall in development mode
uv pip install -e .
```

## 📚 Next Steps

1. Read the full [README.md](../README.md)
2. Explore example notebooks in `notebooks/exploratory/`
3. Review the [API documentation](API.md)
4. Check out research best practices in [RESEARCH_NOTES.md](RESEARCH_NOTES.md)

## 💡 Tips for Success

- **Start simple**: Begin with a baseline model before complex architectures
- **Validate early**: Run on small data subset first
- **Document everything**: Future you will thank present you
- **Ask for help**: Discuss with your advisor/team regularly
- **Iterate quickly**: Fast feedback loops lead to better research

---

Happy researching! 🎓🔬
