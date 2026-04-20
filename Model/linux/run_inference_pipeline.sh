#!/bin/bash
# DepressoSpeech - Full Inference Pipeline (Linux)
# Starts the API server in background, then runs a health check
set -e

echo "=========================================="
echo "  DepressoSpeech - Inference Pipeline"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

CONFIG=${CONFIG:-"configs/inference_config.yaml"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}

# Check that all artifacts exist
MISSING=0
for artifact in "checkpoints/best_model.pt" "checkpoints/scalers/feature_scalers.pkl" "checkpoints/scalers/pca_reducer.pkl"; do
    if [ ! -f "$artifact" ]; then
        echo "ERROR: Missing artifact: $artifact"
        MISSING=1
    fi
done
if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Run the training pipeline first: bash linux/run_training_pipeline.sh"
    exit 1
fi

echo "All model artifacts found."
echo ""
echo "Starting API server on $HOST:$PORT ..."
echo "Server will keep running. Use Ctrl+C to stop."
echo ""
echo "Test with:"
echo "  curl http://$HOST:$PORT/health"
echo "  curl -X POST -F 'file=@audio.wav' http://$HOST:$PORT/predict"
echo "------------------------------------------"

python3 scripts/serve.py --host "$HOST" --port "$PORT" --config "$CONFIG" --log-level info
