#!/usr/bin/env python3
"""
[LAYER_START] Session 8: Inference Entry Point
End-to-end prediction: raw audio → PHQ-8 depression severity score.

Usage:
    python scripts/predict.py --audio path/to/audio.wav
    python scripts/predict.py --audio path/to/audio.wav --config configs/inference_config.yaml
    python scripts/predict.py --audio-dir data/test_audio/ --output results.csv
"""

import sys
import argparse
import logging
import csv
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import InferencePipeline
from src.utils import setup_logging as _setup_logging


def load_config(config_path: str) -> dict:
    """Load inference config YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_pipeline(config: dict) -> InferencePipeline:
    """Build InferencePipeline from config."""
    artifacts = config.get('artifacts', {})
    model_cfg = config.get('model', {})
    inference_cfg = config.get('inference', {})

    # Build audio preprocessing config from the config file
    audio_config = {}
    if 'audio' in config:
        audio_config['audio'] = config['audio']
    if 'preprocessing' in config:
        audio_config['preprocessing'] = config['preprocessing']

    return InferencePipeline(
        model_path=artifacts.get('model_path', 'checkpoints/best_model.pt'),
        normalizer_path=artifacts.get('normalizer_path'),
        pca_path=artifacts.get('pca_path'),
        audio_config=audio_config or None,
        model_config=model_cfg or None,
        device=inference_cfg.get('device', 'auto'),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Depression severity prediction from audio"
    )
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Path to a single audio file",
    )
    parser.add_argument(
        "--audio-dir", type=str, default=None,
        help="Directory of audio files for batch inference",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/inference_config.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path for batch results (default: stdout)",
    )
    parser.add_argument(
        "--extensions", nargs="+",
        default=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        help="Audio file extensions to include in batch mode",
    )
    args = parser.parse_args()

    if args.audio is None and args.audio_dir is None:
        parser.error("Provide --audio (single file) or --audio-dir (batch)")

    log_path = _setup_logging(
        task_name="inference",
        log_dir="logs",
        console_level="INFO",
        file_level="DEBUG",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("[SESSION 8] Depression Severity Inference")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 60)

    # Load config and build pipeline
    config = load_config(args.config)
    pipeline = build_pipeline(config)

    if args.audio:
        # --- Single file inference ---
        result = pipeline.predict_from_audio(
            audio_path=args.audio,
            participant_id=Path(args.audio).stem,
        )

        print()
        print("=" * 40)
        print(f"  File:     {Path(args.audio).name}")
        print(f"  PHQ-8:    {result.phq8_score}")
        print(f"  Severity: {result.severity}")
        print(f"  Chunks:   {result.num_chunks}")
        print("=" * 40)

    else:
        # --- Batch inference ---
        audio_dir = Path(args.audio_dir)
        if not audio_dir.is_dir():
            logger.error(f"Directory not found: {audio_dir}")
            sys.exit(1)

        audio_files = sorted([
            f for f in audio_dir.iterdir()
            if f.suffix.lower() in args.extensions
        ])

        if not audio_files:
            logger.error(f"No audio files found in {audio_dir}")
            sys.exit(1)

        logger.info(f"[INFERENCE_PATH] Found {len(audio_files)} audio files")

        results = pipeline.predict_batch(
            audio_paths=[str(f) for f in audio_files],
        )

        # Output results
        rows = [
            {
                "participant_id": r.participant_id,
                "phq8_score": r.phq8_score,
                "severity": r.severity,
                "num_chunks": r.num_chunks,
            }
            for r in results
        ]

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"[INFERENCE_PATH] Results saved to {output_path}")
        else:
            # Print to stdout as table
            print()
            print(f"{'ID':<15} {'PHQ-8':>6} {'Severity':<22} {'Chunks':>6}")
            print("-" * 55)
            for r in rows:
                print(
                    f"{r['participant_id']:<15} "
                    f"{r['phq8_score']:>6.1f} "
                    f"{r['severity']:<22} "
                    f"{r['num_chunks']:>6}"
                )

        logger.info("=" * 60)
        logger.info(f"[SESSION 8] Batch inference complete: {len(results)} files")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
