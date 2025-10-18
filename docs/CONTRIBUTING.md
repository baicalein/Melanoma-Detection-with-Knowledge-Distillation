# Contributing Guidelines

Thank you for your interest in contributing to this research project! This document provides guidelines for collaboration and code contributions.

## 🤝 How to Contribute

### For Team Members

1. **Create a feature branch**
   ```bash
   git checkout -b feature/descriptive-name
   ```

2. **Make your changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   make check-all  # Runs format, lint, and tests
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/descriptive-name
   ```

## 📝 Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters
- **Formatting**: Use `black` with default settings
- **Import sorting**: Use `isort` with black profile
- **Type hints**: Required for function signatures

### Code Organization

```python
# Standard library imports
import os
from pathlib import Path
from typing import List, Optional

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# Local imports
from deep_learning_final_project.utils import set_seed
```

### Docstring Format

Use Google-style docstrings:

```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10
) -> dict:
    """
    Train a PyTorch model.

    This function trains the provided model using the given dataloader
    for the specified number of epochs.

    Args:
        model: PyTorch model to train
        dataloader: DataLoader providing training data
        epochs: Number of training epochs

    Returns:
        Dictionary containing training metrics

    Raises:
        ValueError: If epochs is negative

    Example:
        >>> model = SimpleCNN()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> metrics = train_model(model, loader, epochs=10)
    """
    pass
```

## 🧪 Testing

### Writing Tests

- Place tests in `tests/` directory
- Mirror the structure of `src/`
- Name test files `test_*.py`
- Use descriptive test names

```python
def test_model_forward_pass_with_valid_input() -> None:
    """Test that model produces correct output shape."""
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    assert output.shape == (4, 10)
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_models.py -v

# Run tests with coverage
pytest --cov=src --cov-report=html
```

## 📦 Adding Dependencies

### Using uv

```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Add with version constraint
uv add "package-name>=1.0.0,<2.0.0"
```

### Updating pyproject.toml

After adding dependencies, commit both `pyproject.toml` and `uv.lock`:

```bash
git add pyproject.toml uv.lock
git commit -m "Add package-name dependency"
```

## 🔍 Code Review Process

### What We Look For

1. **Correctness**: Does the code work as intended?
2. **Tests**: Are there tests for new functionality?
3. **Documentation**: Are docstrings and comments clear?
4. **Style**: Does it follow our coding standards?
5. **Performance**: Are there any obvious inefficiencies?

### Review Checklist

- [ ] Code runs without errors
- [ ] Tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Code is formatted (`make format`)
- [ ] Documentation is updated
- [ ] Commit messages are clear

## 📊 Experiment Guidelines

### Documenting Experiments

1. **Create a config file**
   ```yaml
   # configs/experiments/exp_XXX_description.yaml
   ```

2. **Update experiment log**
   Add entry to `docs/experiments.md`

3. **Save results**
   ```
   results/
   ├── figures/exp_XXX_*.png
   ├── metrics/exp_XXX_metrics.json
   └── tables/exp_XXX_results.csv
   ```

### Reproducibility Requirements

- Set and document random seeds
- Save exact configuration used
- Record git commit hash
- Document hardware specifications
- Note any manual interventions

## 🐛 Reporting Issues

### Bug Reports

Include:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Error messages or logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered
- Impact on existing code

## 💡 Best Practices

### Git Workflow

```bash
# Start with latest main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-feature

# Make small, focused commits
git commit -m "Add function X"
git commit -m "Add tests for X"
git commit -m "Update docs for X"

# Push and open PR
git push origin feature/my-feature
```

### Commit Messages

Format:
```
Brief description (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")
```

Examples:
- ✅ "Add attention mechanism to CNN model"
- ✅ "Fix data loading bug in ImageDataset"
- ✅ "Update training script with early stopping"
- ❌ "fixed stuff"
- ❌ "WIP"

## 🔐 Security

- Never commit API keys or secrets
- Use `.env` files for sensitive information
- Add `.env` to `.gitignore`
- Use environment variables in configs

## 📚 Resources

### Tools Documentation

- [uv](https://github.com/astral-sh/uv)
- [PyTorch](https://pytorch.org/docs/)
- [pytest](https://docs.pytest.org/)
- [ruff](https://github.com/astral-sh/ruff)

### Python Style Guides

- [PEP 8](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

## 🙋 Questions?

If you have questions about contributing:

1. Check existing documentation
2. Ask in team meetings
3. Open an issue for discussion
4. Contact the project maintainer

---

Thank you for contributing to this research project! 🎉
