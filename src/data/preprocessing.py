"""Audio preprocessing utilities for the DepressoSpeech pipeline.

Handles loading, resampling, silence trimming, and normalization of
interview audio prior to feature extraction.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_audio(
    file_path: str,
    sr: Optional[int] = None,
) -> Tuple[NDArray[np.float32], int]:
    """Load an audio file from disk.

    Args:
        file_path: Path to the audio file.
        sr: Target sample rate.  If None, the native rate is preserved.

    Returns:
        Tuple of (audio signal as 1-D float32 array, sample rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sample_rate = librosa.load(str(path), sr=sr, mono=True)
    logger.info(
        "Loaded %s  (duration=%.2fs, sr=%d).",
        path.name,
        len(audio) / sample_rate,
        sample_rate,
    )
    return audio, sample_rate


def resample_audio(
    audio: NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    """Resample audio to a target sample rate.

    Args:
        audio: Input audio signal.
        orig_sr: Original sample rate of the audio.
        target_sr: Desired sample rate.

    Returns:
        Resampled audio signal.
    """
    if orig_sr == target_sr:
        logger.debug("Sample rate already at %d Hz, skipping resample.", target_sr)
        return audio

    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    logger.info("Resampled audio from %d Hz to %d Hz.", orig_sr, target_sr)
    return resampled


def trim_silence(
    audio: NDArray[np.float32],
    top_db: int = 20,
) -> NDArray[np.float32]:
    """Trim leading and trailing silence from audio.

    Args:
        audio: Input audio signal.
        top_db: Threshold (in dB) below the peak amplitude to consider as silence.

    Returns:
        Trimmed audio signal.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    removed = len(audio) - len(trimmed)
    logger.info("Trimmed %d samples of silence (top_db=%d).", removed, top_db)
    return trimmed


def normalize_audio(
    audio: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Peak-normalize audio to the range [-1, 1].

    Args:
        audio: Input audio signal.

    Returns:
        Normalized audio signal.
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        logger.warning("Audio signal is silent, skipping normalization.")
        return audio

    normalized = audio / peak
    logger.debug("Normalized audio (peak was %.6f).", peak)
    return normalized


def preprocess_audio(
    file_path: str,
    target_sr: int = 16000,
    top_db: int = 20,
) -> Tuple[NDArray[np.float32], int]:
    """Full preprocessing pipeline: load → resample → trim → normalize.

    Args:
        file_path: Path to the audio file.
        target_sr: Desired sample rate for output.
        top_db: Silence trimming threshold in dB.

    Returns:
        Tuple of (preprocessed audio signal, sample rate).
    """
    audio, sr = load_audio(file_path, sr=None)
    audio = resample_audio(audio, orig_sr=sr, target_sr=target_sr)
    audio = trim_silence(audio, top_db=top_db)
    audio = normalize_audio(audio)
    logger.info("Preprocessing complete for %s.", file_path)
    return audio, target_sr
