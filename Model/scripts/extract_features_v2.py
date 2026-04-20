#!/usr/bin/env python3
"""
Multimodal V2 Feature Extraction: HuBERT + SBERT with Fixed-Window Segmentation.

Key changes from v1 (extract_features.py):
  1. SEGMENTATION FIX: Uses fixed 5s windows (matching inference) instead of
     variable-length transcript segments. Eliminates train-inference distribution shift.
  2. AUDIO UPGRADE: HuBERT Base (768-dim) replaces MFCC(120) + eGeMAPS(88).
     Single learned representation captures prosody, pauses, vocal quality.
  3. AUDIO QUALITY: Per-segment quality score stored for gated fusion.
  4. TEXT SOURCE: Whisper Base transcription per 5s chunk (same as inference).

Output per participant: {pid}_training_v2.npz with keys:
  - hubert: (N, 768) — HuBERT embeddings
  - text_embeddings: (N, 384) — SBERT embeddings
  - audio_quality: (N,) — per-segment quality scores [0, 1]
  - source: "training_v2"

Usage:
    python scripts/extract_features_v2.py
    python scripts/extract_features_v2.py --data-dir data/raw --output-dir data/features_v2
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hubert_extractor import HuBERTExtractor
from src.features.text_extractor import TextExtractor
from src.features.audio_quality import AudioQualityScorer
from src.utils import setup_logging

SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0      # seconds — matches inference chunker
CHUNK_OVERLAP = 0.25      # 25% overlap
MIN_CHUNK_DURATION = 2.0  # seconds


def chunk_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list:
    """
    Fixed-window chunking with overlap — identical to inference AudioChunker.

    Returns list of (audio_chunk, start_time, end_time) tuples.
    """
    chunk_samples = int(CHUNK_DURATION * sr)
    hop_samples = int(CHUNK_DURATION * (1 - CHUNK_OVERLAP) * sr)
    min_samples = int(MIN_CHUNK_DURATION * sr)

    total_samples = len(audio)
    if total_samples < min_samples:
        return [(audio, 0.0, total_samples / sr)]

    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]

        # Skip too-short final chunks
        if len(chunk) < min_samples and start > 0:
            break

        # Pad short final chunk
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

        start_time = start / sr
        end_time = min(end, total_samples) / sr
        chunks.append((chunk, start_time, end_time))

        start += hop_samples

    return chunks


def extract_for_participant(
    pid: str,
    data_dir: Path,
    hubert_ext: HuBERTExtractor,
    text_ext: TextExtractor,
    quality_scorer: AudioQualityScorer,
    output_dir: Path,
) -> bool:
    """
    Extract HuBERT + SBERT features for one participant using fixed-window segmentation.

    Pipeline:
        1. Load audio WAV
        2. Chunk audio into fixed 5s windows (matching inference)
        3. Extract HuBERT embeddings per chunk → (N, 768)
        4. Transcribe each chunk with Whisper → text
        5. Encode text with SBERT → (N, 384)
        6. Score audio quality per chunk → (N,)
        7. Save to NPZ
    """
    logger = logging.getLogger(__name__)

    pid_dir = data_dir / pid
    if not pid_dir.is_dir():
        logger.warning(f"[VALIDATION_CHECK] Participant {pid}: directory not found")
        return False

    # Find audio file
    audio_path = None
    for child in pid_dir.iterdir():
        if child.is_file() and child.suffix.lower() == '.wav':
            name_lower = child.name.lower()
            if 'audio' in name_lower:
                audio_path = child
                break
    if audio_path is None:
        logger.warning(f"[VALIDATION_CHECK] Participant {pid}: no audio WAV found")
        return False

    try:
        # Step 1: Load audio
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        logger.info(
            f"[DATA_FLOW] Participant {pid}: audio={len(audio)/SAMPLE_RATE:.1f}s"
        )

        # Step 2: Fixed-window chunking
        chunks = chunk_audio(audio, sr=SAMPLE_RATE)
        if not chunks:
            logger.warning(f"[VALIDATION_CHECK] Participant {pid}: no chunks produced")
            return False

        audio_segments = [c[0] for c in chunks]
        n_chunks = len(audio_segments)

        # Step 3: HuBERT embeddings
        hubert = hubert_ext.extract_from_audio(audio_segments, sr=SAMPLE_RATE)

        # Step 4 + 5: Whisper transcription → SBERT encoding
        audio_array = np.stack(audio_segments)  # (N, samples)
        start_times = [c[1] for c in chunks]
        text_embeddings = text_ext.extract_from_audio(
            audio_array, sr=SAMPLE_RATE, chunk_start_times=start_times
        )

        # Step 6: Audio quality scores
        audio_quality = quality_scorer.score_segments(audio_segments)

        # Verify shapes
        assert hubert.shape == (n_chunks, 768), f"hubert shape: {hubert.shape}"
        assert text_embeddings.shape == (n_chunks, 384), f"text shape: {text_embeddings.shape}"
        assert audio_quality.shape == (n_chunks,), f"quality shape: {audio_quality.shape}"

        # Step 7: Save
        output_path = output_dir / f"{pid}_training_v2.npz"
        np.savez_compressed(
            output_path,
            hubert=hubert,
            text_embeddings=text_embeddings,
            audio_quality=audio_quality,
            source="training_v2",
        )

        logger.info(
            f"[DATA_FLOW] Participant {pid}: saved {n_chunks} chunks "
            f"(hubert={hubert.shape}, text={text_embeddings.shape}, "
            f"quality_mean={audio_quality.mean():.3f})"
        )
        return True

    except Exception as e:
        logger.error(f"[DEBUG_POINT] Participant {pid} failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract V2 multimodal features (HuBERT + SBERT, fixed windows)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Directory containing raw per-participant data",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/features_v2",
        help="Output directory for V2 feature NPZ files",
    )
    parser.add_argument(
        "--splits-dir", type=str, default="data/splits",
        help="Directory containing split CSVs",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "dev", "test"],
        help="Which splits to process",
    )
    parser.add_argument(
        "--id-column", type=str, default="Participant_ID",
        help="Column name for participant ID in split CSVs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if V2 features already exist",
    )
    args = parser.parse_args()

    log_path = setup_logging(
        task_name="feature_extraction_v2",
        log_dir="logs",
        console_level="INFO",
        file_level="DEBUG",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("[V2] Multimodal Feature Extraction: HuBERT + SBERT")
    logger.info(f"  Segmentation: fixed {CHUNK_DURATION}s windows, {CHUNK_OVERLAP*100:.0f}% overlap")
    logger.info(f"  Audio: HuBERT Base (768-dim)")
    logger.info(f"  Text: Whisper Base → SBERT (384-dim)")
    logger.info(f"  Log file: {log_path}")
    logger.info("=" * 60)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)

    # Initialize extractors
    hubert_ext = HuBERTExtractor()
    text_ext = TextExtractor()
    quality_scorer = AudioQualityScorer()

    # Collect PIDs from splits
    split_files = {
        "train": "train_split.csv",
        "dev": "dev_split.csv",
        "test": "test_split.csv",
    }

    all_pids = []
    for split_name in args.splits:
        csv_name = split_files.get(split_name)
        if csv_name is None:
            continue
        csv_path = splits_dir / csv_name
        if not csv_path.exists():
            logger.warning(f"[VALIDATION_CHECK] Split CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        pids = [str(int(row[args.id_column])) for _, row in df.iterrows()]
        all_pids.extend(pids)
        logger.info(f"[DATA_FLOW] Split '{split_name}': {len(pids)} participants")

    all_pids = sorted(set(all_pids))
    logger.info(f"[DATA_FLOW] Total unique participants: {len(all_pids)}")

    success = 0
    failed = 0
    skipped = 0

    for i, pid in enumerate(all_pids, 1):
        # Skip if already extracted
        output_path = output_dir / f"{pid}_training_v2.npz"
        if not args.force and output_path.exists():
            logger.debug(f"[DATA_FLOW] Participant {pid}: already exists, skipping")
            skipped += 1
            continue

        ok = extract_for_participant(
            pid=pid,
            data_dir=data_dir,
            hubert_ext=hubert_ext,
            text_ext=text_ext,
            quality_scorer=quality_scorer,
            output_dir=output_dir,
        )

        if ok:
            success += 1
        else:
            failed += 1

        if i % 20 == 0:
            logger.info(f"[DATA_FLOW] Progress: {i}/{len(all_pids)}")

    logger.info("=" * 60)
    logger.info(f"[V2] Extraction Complete")
    logger.info(f"  Success: {success}")
    logger.info(f"  Skipped: {skipped} (already exist)")
    logger.info(f"  Failed:  {failed}")
    logger.info(f"  Store:   {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
