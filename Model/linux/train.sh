#!/bin/bash
# DepressoSpeech - Training Pipeline (Linux)
# Trains the DepressionModel (MLP + BiGRU + Attention)
set -e

echo "=========================================="
echo "  DepressoSpeech - Training Pipeline"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

CONFIG=${CONFIG:-"configs/training_config.yaml"}
FEATURE_DIR=${FEATURE_DIR:-"data/features"}
USE_PRECOMPUTED_PCA=${USE_PRECOMPUTED_PCA:-""}

echo "Training Configuration:"
echo "  Config:      $CONFIG"
echo "  Feature Dir: $FEATURE_DIR"
echo ""

# Check for GPU
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Device: CPU')
"

# Ensure checkpoint dirs exist
mkdir -p checkpoints/scalers

# Build command
CMD="python3 scripts/train.py --config $CONFIG --feature-dir $FEATURE_DIR"
if [ -n "$USE_PRECOMPUTED_PCA" ]; then
    CMD="$CMD --use-precomputed-pca"
    echo "  Using pre-computed PCA: Yes"
fi

echo ""
echo "Starting training..."
echo "------------------------------------------"
$CMD

echo "------------------------------------------"
echo "  Training complete!"
echo "  Artifacts saved:"
echo "    Model:      checkpoints/best_model.pt"
echo "    Scalers:    checkpoints/scalers/feature_scalers.pkl"
echo "    PCA:        checkpoints/scalers/pca_reducer.pkl"
echo "=========================================="
