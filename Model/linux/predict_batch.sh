#!/bin/bash
# DepressoSpeech - Batch Inference (Linux)
# Predicts PHQ-8 scores for all audio files in a directory
set -e

echo "=========================================="
echo "  DepressoSpeech - Batch Inference"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

CONFIG=${CONFIG:-"configs/inference_config.yaml"}
AUDIO_DIR=${1:-""}
OUTPUT_CSV=${2:-""}
EXTENSIONS=${EXTENSIONS:-".wav .mp3 .flac .ogg .m4a"}

if [ -z "$AUDIO_DIR" ]; then
    echo "Usage: bash linux/predict_batch.sh <audio_directory> [output.csv]"
    echo ""
    echo "Example:"
    echo "  bash linux/predict_batch.sh data/test_audio/"
    echo "  bash linux/predict_batch.sh data/test_audio/ results.csv"
    exit 1
fi

if [ ! -d "$AUDIO_DIR" ]; then
    echo "ERROR: Directory not found: $AUDIO_DIR"
    exit 1
fi

if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "ERROR: Model checkpoint not found at checkpoints/best_model.pt"
    echo "Please train the model first: bash linux/train.sh"
    exit 1
fi

echo "Config:     $CONFIG"
echo "Audio Dir:  $AUDIO_DIR"
echo "Extensions: $EXTENSIONS"
[ -n "$OUTPUT_CSV" ] && echo "Output CSV: $OUTPUT_CSV"
echo ""
echo "Running batch inference..."
echo "------------------------------------------"

CMD="python3 scripts/predict.py --audio-dir $AUDIO_DIR --config $CONFIG --extensions $EXTENSIONS"
[ -n "$OUTPUT_CSV" ] && CMD="$CMD --output $OUTPUT_CSV"

$CMD

echo "------------------------------------------"
echo "  Batch inference complete!"
echo "=========================================="
