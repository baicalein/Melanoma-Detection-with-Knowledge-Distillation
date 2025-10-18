# Quick Start Guide

Get started with this deep learning project in 5 minutes!

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

## 📝 Your First Experiment

### 1. Explore the Template Notebook

```bash
jupyter lab notebooks/exploratory/00_example_eda.ipynb
```

### 2. Add Your Data

```bash
# Place your data in the appropriate directory
data/
├── raw/           # Original data here
├── processed/     # Cleaned data here
└── external/      # Reference data here
```

### 3. Configure Your Experiment

Edit `configs/experiments/baseline.yaml` with your settings:

```yaml
experiment:
  name: my_first_experiment
  
model:
  num_classes: 10
  
training:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 100
```

### 4. Train Your Model

```bash
# Option 1: Use the template (modify as needed)
python -m deep_learning_final_project.scripts.train \
    --config configs/experiments/baseline.yaml

# Option 2: Use a Jupyter notebook
jupyter lab
```

## 📚 Key Files to Know

- **`README.md`** - Full project documentation
- **`docs/GETTING_STARTED.md`** - Detailed setup guide
- **`docs/RESEARCH_NOTES.md`** - Academic research best practices
- **`pyproject.toml`** - Project dependencies
- **`.env.example`** - Environment variables template

## 🎯 Common Tasks

### Adding Dependencies

```bash
# Add a package
uv add package-name

# Add a dev dependency
uv add --dev package-name
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_models.py -v

# With coverage
pytest --cov=src
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## 🆘 Troubleshooting

### "Module not found" error
```bash
# Reinstall in editable mode
uv pip install -e .
```

### GPU not detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Tests failing
```bash
# Clear cache and retry
make clean
make test
```

## 📖 Next Steps

1. Read [GETTING_STARTED.md](docs/GETTING_STARTED.md) for detailed workflows
2. Explore [RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) for best practices
3. Check out example code in `src/deep_learning_final_project/`
4. Review test files in `tests/` for usage examples

## 💡 Pro Tips

- **Use `make` commands** - Convenient shortcuts for common tasks
- **Check the tests** - They show how to use each module
- **Read the docstrings** - All functions have detailed documentation
- **Copy the templates** - Adapt example files for your needs
- **Commit frequently** - Small, focused commits are best

## 🎓 For Academic Research

- Set random seeds for reproducibility
- Document all experiments in `docs/experiments.md`
- Save configurations with results
- Use version control for everything except large data files
- Generate figures programmatically for easy updates

---

**Need more help?** Check the full [README.md](README.md) or [GETTING_STARTED.md](docs/GETTING_STARTED.md)

Happy coding! 🚀
