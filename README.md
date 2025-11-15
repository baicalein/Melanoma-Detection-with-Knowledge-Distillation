# Objective

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repo explores high performing knowledge distillation targeting cell phones and edge devices using the HAM1000 melonoma classification dataset.

## Table of Contents

- [Objective](#objective)
  - [Table of Contents](#table-of-contents)
  - [Important Links](#important-links)
  - [Quick Setup](#quick-setup)
  - [Contact](#contact)
  - [Useful commands](#useful-commands)

## Important Links

- [Edstem for discussion](https://edstem.org/us/courses/83711/discussion/6884545)
- [Overleaf working document](https://www.overleaf.com/read/ycgbrjvyyqbx#5162eb)
- [TA Rivanna Guide](https://github.com/JustUnoptimized/ds6050-rivanna)

## Quick Setup

```bash
  make install       - Install dependencies
  make test          - Run tests with coverage
  make lint          - Run linters (ruff, mypy)
  make format        - Format code (black, isort)
  make clean         - Clean build artifacts and cache
  make pre-commit    - Install pre-commit hooks
  make check-all     - Run format, lint, and test
```

## Contact

Ryan Healy (rah5ff) and Angie Yoon

## Useful commands

```bash
## see folders structure
tree -d -I '.git|.venv|venv|__pycache__' -L 3
```
