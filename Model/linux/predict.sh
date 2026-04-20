#!/bin/bash
# DepressoSpeech - Single File Inference (Linux)
# Predicts PHQ-8 score from a single audio file
set -e

echo "=========================================="
echo "  DepressoSpeech - Single File Inference"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

CONFIG=${CONFIG:-"configs/inference_config.yaml"}
AUDIO_FILE=${1:-""}

if [ -z "$AUDIO_FILE" ]; then
    echo "Usage: bash linux/predict.sh <path_to_audio_file>"
    echo ""
    echo "Supported formats: .wav .mp3 .flac .ogg .m4a"
    echo ""
    echo "Example:"
    echo "  bash linux/predict.sh /path/to/interview.wav"
    exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo "ERROR: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Check that model artifacts exist
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "ERROR: Model checkpoint not found at checkpoints/best_model.pt"
    echo "Please train the model first: bash linux/train.sh"
    exit 1
fi

echo "Config:     $CONFIG"
echo "Audio File: $AUDIO_FILE"
echo ""
echo "Running inference..."
echo "------------------------------------------"

python3 scripts/predict.py \
    --audio "$AUDIO_FILE" \
    --config "$CONFIG"

echo "------------------------------------------"
echo "  Inference complete!"
echo "=========================================="
