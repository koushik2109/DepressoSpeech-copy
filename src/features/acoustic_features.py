"""Acoustic feature extraction for the DepressoSpeech pipeline.

Extracts MFCC, pitch (F0), spectral centroid, and energy features
from preprocessed audio signals using librosa, numpy, and scipy.
"""

import logging
from typing import Any, Dict, Optional

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import medfilt

logger = logging.getLogger(__name__)


def extract_mfcc(
    audio: NDArray[np.float32],
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NDArray[np.float32]:
    """Extract Mel-frequency cepstral coefficients.

    Args:
        audio: Preprocessed audio signal.
        sr: Sample rate.
        n_mfcc: Number of MFCCs to extract.
        n_fft: FFT window size.
        hop_length: Hop length in samples.

    Returns:
        MFCC matrix of shape (n_mfcc, n_frames).
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    logger.info("Extracted MFCCs: shape=%s.", mfcc.shape)
    return mfcc


def extract_pitch(
    audio: NDArray[np.float32],
    sr: int,
    fmin: float = 50.0,
    fmax: float = 500.0,
    hop_length: int = 512,
) -> NDArray[np.float32]:
    """Extract fundamental frequency (pitch) contour using pyin.

    Args:
        audio: Preprocessed audio signal.
        sr: Sample rate.
        fmin: Minimum expected frequency in Hz.
        fmax: Maximum expected frequency in Hz.
        hop_length: Hop length in samples.

    Returns:
        1-D array of F0 values per frame (NaN where unvoiced).
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    # Replace NaN with 0 for downstream compatibility
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    logger.info(
        "Extracted pitch: %d frames, %.1f%% voiced.",
        len(f0),
        np.mean(voiced_flag) * 100,
    )
    return f0


def extract_spectral_centroid(
    audio: NDArray[np.float32],
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NDArray[np.float32]:
    """Extract the spectral centroid over time.

    Args:
        audio: Preprocessed audio signal.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length in samples.

    Returns:
        1-D array of spectral centroid values per frame.
    """
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    centroid = centroid.squeeze()
    logger.info("Extracted spectral centroid: %d frames.", len(centroid))
    return centroid


def extract_energy(
    audio: NDArray[np.float32],
    hop_length: int = 512,
    frame_length: int = 2048,
) -> NDArray[np.float32]:
    """Compute short-time energy (RMS) per frame.

    Args:
        audio: Preprocessed audio signal.
        hop_length: Hop length in samples.
        frame_length: Frame length in samples.

    Returns:
        1-D array of RMS energy values per frame.
    """
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )
    energy = rms.squeeze()
    logger.info("Extracted energy: %d frames.", len(energy))
    return energy


def extract_all_features(
    audio: NDArray[np.float32],
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict[str, Any]:
    """Extract the full acoustic feature set from an audio signal.

    Features extracted:
        - mfcc: Mel-frequency cepstral coefficients
        - pitch: Fundamental frequency (F0) contour
        - spectral_centroid: Spectral centroid per frame
        - energy: RMS energy per frame

    Args:
        audio: Preprocessed audio signal.
        sr: Sample rate.
        n_mfcc: Number of MFCCs.
        n_fft: FFT window size.
        hop_length: Hop length in samples.

    Returns:
        Dictionary containing all feature arrays and metadata.
    """
    logger.info("Starting full feature extraction (sr=%d).", sr)

    features: Dict[str, Any] = {
        "mfcc": extract_mfcc(audio, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length),
        "pitch": extract_pitch(audio, sr, hop_length=hop_length),
        "spectral_centroid": extract_spectral_centroid(audio, sr, n_fft=n_fft, hop_length=hop_length),
        "energy": extract_energy(audio, hop_length=hop_length, frame_length=n_fft),
        "metadata": {
            "sample_rate": sr,
            "n_mfcc": n_mfcc,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "duration_samples": len(audio),
        },
    }

    logger.info("Feature extraction complete.")
    return features
