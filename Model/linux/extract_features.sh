#!/bin/bash
# DepressoSpeech - Feature Extraction Pipeline (Linux)
# Extracts eGeMAPS, MFCC, and text embeddings from raw audio
set -e

echo "=========================================="
echo "  DepressoSpeech - Feature Extraction"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

DATA_DIR=${DATA_DIR:-"data/raw"}
OUTPUT_DIR=${OUTPUT_DIR:-"data/features"}
SPLITS_DIR=${SPLITS_DIR:-"data/splits"}
SPLITS=${SPLITS:-"train dev test"}

# Verify data directories
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $DATA_DIR is empty or missing!"
    echo ""
    echo "Expected structure:"
    echo "  $DATA_DIR/<participant_id>/<participant_id>_AUDIO.wav"
    echo ""
    echo "And split CSVs in $SPLITS_DIR/:"
    echo "  train_split.csv, dev_split.csv, test_split.csv"
    exit 1
fi

for split in $SPLITS; do
    if [ ! -f "$SPLITS_DIR/${split}_split.csv" ]; then
        echo "ERROR: $SPLITS_DIR/${split}_split.csv not found!"
        exit 1
    fi
done

echo "Configuration:"
echo "  Data Dir:    $DATA_DIR"
echo "  Output Dir:  $OUTPUT_DIR"
echo "  Splits Dir:  $SPLITS_DIR"
echo "  Splits:      $SPLITS"
echo ""

echo "Starting feature extraction..."
echo "------------------------------------------"
python3 scripts/extract_features.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --splits $SPLITS \
    --splits-dir "$SPLITS_DIR" \
    --id-column "Participant_ID"

echo "------------------------------------------"
echo "  Feature extraction complete!"
echo "  Features saved to: $OUTPUT_DIR/"
echo "=========================================="
