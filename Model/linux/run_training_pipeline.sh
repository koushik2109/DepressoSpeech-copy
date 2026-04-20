#!/bin/bash
# DepressoSpeech - Full Training Pipeline (Linux)
# Runs: setup → extract features → train
set -e

echo "=========================================="
echo "  DepressoSpeech - Full Training Pipeline"
echo "=========================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Step 1: Setup
echo "[Step 1/3] Setting up environment..."
bash linux/setup.sh

# Step 2: Extract Features
echo ""
echo "[Step 2/3] Extracting features..."
bash linux/extract_features.sh

# Step 3: Training
echo ""
echo "[Step 3/3] Training model..."
bash linux/train.sh

echo ""
echo "=========================================="
echo "  Full training pipeline complete!"
echo ""
echo "  Next steps:"
echo "    Single prediction:  bash linux/predict.sh <audio.wav>"
echo "    Batch prediction:   bash linux/predict_batch.sh <audio_dir/>"
echo "    Start API server:   bash linux/serve.sh"
echo "=========================================="
