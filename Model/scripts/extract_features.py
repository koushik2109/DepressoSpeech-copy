#!/usr/bin/env python3
"""
[LAYER_START] Session 7: Feature Extraction Entry Point
Extracts eGeMAPS, MFCC, and text features per participant from audio + transcript.

Uses audio-based extraction (OpenSMILE + librosa) aligned with transcript
timestamps, ensuring exact training-inference feature parity.

Must be run BEFORE scripts/train.py.

Usage:
    python scripts/extract_features.py
    python scripts/extract_features.py --data-dir data/raw --output-dir data/features
    python scripts/extract_features.py --splits train dev test
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    EgemapsExtractor,
    MfccExtractor,
    TextExtractor,
    FeatureStore,
)
from src.utils import setup_logging

# Sample rate for audio loading (matches inference pipeline)
SAMPLE_RATE = 16000


def load_transcript(csv_path: Path, max_utterance_seconds: float = 120.0) -> list:
    """
    Load transcript CSV and return list of (start_time, end_time, text) tuples.

    Handles DAIC-WOZ format: Start_Time, End_Time, Text[, Confidence]
    The Confidence column is ignored (Whisper does not produce it).

    Filters out erroneous entries that span most of the audio (a known
    data artifact where the last CSV row covers the entire duration).
    """
    logger = logging.getLogger(__name__)
    df = pd.read_csv(csv_path)

    # Detect column names (case-insensitive)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('start_time', 'start'):
            col_map['start'] = col
        elif cl in ('end_time', 'end', 'stop_time'):
            col_map['end'] = col
        elif cl in ('text', 'value', 'transcript', 'transcription', 'utterance'):
            col_map['text'] = col

    if 'start' not in col_map or 'end' not in col_map or 'text' not in col_map:
        raise ValueError(
            f"Transcript CSV {csv_path.name} missing required columns. "
            f"Found: {list(df.columns)}, need: Start_Time, End_Time, Text"
        )

    segments = []
    dropped = 0
    for _, row in df.iterrows():
        start = float(row[col_map['start']])
        end = float(row[col_map['end']])
        text = str(row[col_map['text']]).strip() if pd.notna(row[col_map['text']]) else ""
        duration = end - start
        if duration <= 0 or not text:
            continue
        if duration > max_utterance_seconds:
            dropped += 1
            continue
        segments.append((start, end, text))

    # Sort by start time for temporal order
    segments.sort(key=lambda x: x[0])

    if dropped > 0:
        logger.info(
            f"[DATA_FLOW] {csv_path.name}: dropped {dropped} entries "
            f"exceeding {max_utterance_seconds}s (known data artifact)"
        )

    return segments


def load_audio(audio_path: Path) -> np.ndarray:
    """Load audio file at SAMPLE_RATE using librosa."""
    import librosa
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return audio


def segment_audio(audio: np.ndarray, segments: list, sr: int = SAMPLE_RATE) -> list:
    """
    Slice audio into per-utterance segments using transcript timestamps.

    Args:
        audio: Full audio waveform (1-D float32)
        segments: List of (start_time, end_time, text) from load_transcript()
        sr: Sample rate

    Returns:
        List of 1-D numpy arrays (variable-length audio segments)
    """
    audio_segments = []
    total_samples = len(audio)

    for start, end, _ in segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        # Clamp to audio bounds
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))

        chunk = audio[start_sample:end_sample]
        audio_segments.append(chunk)

    return audio_segments


def extract_for_participant(
    pid: str,
    data_dir: Path,
    egemaps_ext: EgemapsExtractor,
    mfcc_ext: MfccExtractor,
    text_ext: TextExtractor,
    store: FeatureStore,
) -> bool:
    """
    Extract all features for one participant using audio + transcript.

    Pipeline per participant:
        1. Load transcript → (start, end, text) per utterance
        2. Load audio WAV
        3. Slice audio per utterance timestamps
        4. Extract eGeMAPS functionals (OpenSMILE) → (N, 88)
        5. Extract 40 MFCCs + deltas (librosa) → (N, 120)
        6. Encode text with SBERT → (N, 384)
        7. Save via FeatureStore

    Returns True if extraction succeeded.
    """
    logger = logging.getLogger(__name__)

    # --- Find files ---
    pid_dir = data_dir / pid
    if not pid_dir.is_dir():
        logger.warning(f"[VALIDATION_CHECK] Participant {pid}: directory not found")
        return False

    audio_path = _find_file_by_keyword(pid_dir, ['audio', 'AUDIO'], suffix='.wav')
    transcript_path = _find_file_by_keyword(pid_dir, ['transcript', 'Transcript'], suffix='.csv')

    if audio_path is None:
        logger.warning(f"[VALIDATION_CHECK] Participant {pid}: no audio WAV found")
        return False
    if transcript_path is None:
        logger.warning(f"[VALIDATION_CHECK] Participant {pid}: no transcript CSV found")
        return False

    try:
        # Step 1: Load transcript
        segments = load_transcript(transcript_path)
        if not segments:
            logger.warning(f"[VALIDATION_CHECK] Participant {pid}: empty transcript")
            return False

        # Step 2: Load audio
        audio = load_audio(audio_path)
        logger.info(
            f"[DATA_FLOW] Participant {pid}: audio={len(audio)/SAMPLE_RATE:.1f}s, "
            f"utterances={len(segments)}"
        )

        # Step 3: Segment audio by transcript timestamps
        audio_segments = segment_audio(audio, segments, sr=SAMPLE_RATE)

        # Step 4: Extract eGeMAPS (88-dim per segment)
        egemaps = egemaps_ext.extract_from_audio(audio_segments, sr=SAMPLE_RATE)

        # Step 5: Extract MFCC (120-dim per segment)
        mfcc = mfcc_ext.extract_from_audio(audio_segments, sr=SAMPLE_RATE)

        # Step 6: Encode text with SBERT (384-dim per segment)
        texts = [text for _, _, text in segments]
        text_ext._init_sbert()
        text_emb = text_ext._sbert.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Verify alignment
        n_segments = len(segments)
        assert egemaps.shape == (n_segments, 88), f"egemaps shape mismatch: {egemaps.shape}"
        assert mfcc.shape == (n_segments, 120), f"mfcc shape mismatch: {mfcc.shape}"
        assert text_emb.shape == (n_segments, 384), f"text shape mismatch: {text_emb.shape}"

        # Step 7: Save via FeatureStore
        store.save(
            participant_id=pid,
            egemaps=egemaps,
            mfcc=mfcc,
            text_embeddings=text_emb,
            source="training",
        )

        logger.info(
            f"[DATA_FLOW] Participant {pid}: saved {n_segments} utterances "
            f"(egemaps={egemaps.shape}, mfcc={mfcc.shape}, text={text_emb.shape})"
        )
        return True

    except Exception as e:
        logger.error(f"[DEBUG_POINT] Participant {pid} failed: {e}", exc_info=True)
        return False


def _find_file_by_keyword(directory: Path, keywords: list, suffix: str = '.csv') -> Path:
    """Find a file in directory whose name contains any of the keywords."""
    for child in directory.iterdir():
        if child.is_file() and child.suffix.lower() == suffix.lower():
            name_lower = child.name.lower()
            for kw in keywords:
                if kw.lower() in name_lower:
                    return child
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from audio + transcript per participant"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Directory containing raw per-participant data (audio + transcript)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/features",
        help="Output directory for FeatureStore .npz files",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "dev", "test"],
        help="Which splits to process (default: train dev test)",
    )
    parser.add_argument(
        "--splits-dir", type=str, default="data/splits",
        help="Directory containing split CSVs",
    )
    parser.add_argument(
        "--id-column", type=str, default="Participant_ID",
        help="Column name for participant ID in split CSVs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if features already exist",
    )
    args = parser.parse_args()

    log_path = setup_logging(
        task_name="feature_extraction",
        log_dir="logs",
        console_level="INFO",
        file_level="DEBUG",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("[SESSION 7] Audio-Based Feature Extraction Pipeline")
    logger.info(f"  Extraction method: audio (OpenSMILE + librosa + SBERT)")
    logger.info(f"  Log file: {log_path}")
    logger.info("=" * 60)

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_dir)
    store = FeatureStore(store_dir=args.output_dir)

    # Initialize extractors
    egemaps_ext = EgemapsExtractor()
    mfcc_ext = MfccExtractor()
    text_ext = TextExtractor()

    # Collect all participant IDs from requested splits
    split_files = {
        "train": "train_split.csv",
        "dev": "dev_split.csv",
        "test": "test_split.csv",
    }

    all_pids = []
    for split_name in args.splits:
        csv_name = split_files.get(split_name)
        if csv_name is None:
            logger.warning(f"[VALIDATION_CHECK] Unknown split: {split_name}")
            continue

        csv_path = splits_dir / csv_name
        if not csv_path.exists():
            logger.warning(f"[VALIDATION_CHECK] Split CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        pids = [str(int(row[args.id_column])) for _, row in df.iterrows()]
        all_pids.extend(pids)
        logger.info(f"[DATA_FLOW] Split '{split_name}': {len(pids)} participants")

    # Deduplicate
    all_pids = sorted(set(all_pids))
    logger.info(f"[DATA_FLOW] Total unique participants: {len(all_pids)}")

    # Extract features
    success = 0
    failed = 0
    skipped = 0

    for i, pid in enumerate(all_pids, 1):
        # Skip if already extracted (unless --force)
        if not args.force and store.exists(pid, source="training"):
            logger.debug(f"[DATA_FLOW] Participant {pid}: already exists, skipping")
            skipped += 1
            continue

        ok = extract_for_participant(
            pid=pid,
            data_dir=data_dir,
            egemaps_ext=egemaps_ext,
            mfcc_ext=mfcc_ext,
            text_ext=text_ext,
            store=store,
        )

        if ok:
            success += 1
        else:
            failed += 1

        if i % 20 == 0:
            logger.info(f"[DATA_FLOW] Progress: {i}/{len(all_pids)}")

    logger.info("=" * 60)
    logger.info(f"[SESSION 7] Extraction Complete")
    logger.info(f"  Success: {success}")
    logger.info(f"  Skipped: {skipped} (already exist)")
    logger.info(f"  Failed:  {failed}")
    logger.info(f"  Store:   {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
