# Makefile for Deep Learning Final Project
# Provides convenient commands for common development tasks

.PHONY: help install install-dev test lint format clean run-notebook train docs

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linters (ruff, mypy)"
	@echo "  make format        - Format code (black, isort)"
	@echo "  make clean         - Clean build artifacts and cache"
	@echo "  make run-notebook  - Start Jupyter Lab"
	@echo "  make train         - Run example training script"
	@echo "  make docs          - Build documentation"
	@echo "  make pre-commit    - Install pre-commit hooks"
	@echo "  make check-all     - Run format, lint, and test"

# Installation
install:
	uv sync

install-dev:
	uv sync --extra dev

# Testing
test:
	pytest tests/ -v --cov=src/deep_learning_final_project --cov-report=html --cov-report=term-missing

test-quick:
	pytest tests/ -v

# Code quality
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# Pre-commit
pre-commit:
	pre-commit install
	@echo "Pre-commit hooks installed!"

pre-commit-run:
	pre-commit run --all-files

# Clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ htmlcov/ .coverage

# Development
run-notebook:
	jupyter lab

train:
	@echo "Example training command:"
	@echo "python -m deep_learning_final_project.scripts.train --config configs/experiments/baseline.yaml"

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html

# Combined checks
check-all: format lint test
	@echo "All checks passed!"

# Project structure
tree:
	tree -I '__pycache__|*.pyc|.git|.venv|*.egg-info|.pytest_cache|.ruff_cache' -L 3
