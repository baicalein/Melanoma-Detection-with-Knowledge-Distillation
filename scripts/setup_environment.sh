#!/bin/bash
# Setup script for the deep learning project environment
# Run this script after cloning the repository

set -e  # Exit on error

echo "🚀 Setting up Deep Learning Project Environment"
echo "================================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed."
    echo "Installing uv..."
    pip install uv
fi

echo "✅ uv is installed"

# Create virtual environment and install dependencies
echo ""
echo "📦 Installing dependencies..."
uv sync --extra dev

echo ""
echo "✅ Dependencies installed"

# Install pre-commit hooks
echo ""
echo "🔧 Setting up pre-commit hooks..."
source .venv/bin/activate 2>/dev/null || true
pre-commit install || echo "⚠️  Could not install pre-commit hooks (optional)"

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,external,interim}
mkdir -p logs/tensorboard
mkdir -p checkpoints
mkdir -p results/{figures,tables,metrics}

echo ""
echo "✅ Project directories created"

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your settings"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Edit .env with your settings"
echo ""
echo "  3. Start exploring:"
echo "     jupyter lab"
echo ""
echo "  4. Run tests:"
echo "     make test"
echo ""
echo "Happy researching! 🎓"
