#!/bin/bash
# DepressoSpeech - Environment Setup (Linux)
set -e

echo "=========================================="
echo "  DepressoSpeech - Environment Setup"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Check Python version
python3 --version || { echo "Python3 not found. Please install Python 3.8+"; exit 1; }

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[3/4] Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "[4/4] Creating project directories..."
mkdir -p data/raw
mkdir -p data/features
mkdir -p data/splits
mkdir -p checkpoints/scalers
mkdir -p logs

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Activate env: source .venv/bin/activate"
echo "=========================================="
