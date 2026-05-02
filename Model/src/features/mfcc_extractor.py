"""
[LAYER_START] Feature Extraction - MFCC Extractor
Extracts 40 MFCCs + deltas (120-dim) from audio via librosa.

Both training and inference use the same audio-based extraction path,
ensuring exact feature parity.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union, List, Sequence

from src.features.constants import MFCC_DIM

logger = logging.getLogger(__name__)

# Minimum segment duration for reliable MFCC + delta computation
_MIN_SEGMENT_SECONDS = 0.25


class MfccExtractor:
    """
    MFCC feature extractor using librosa.

    Both training and inference extract 40 MFCCs + delta + delta-delta
    from audio segments, then take the mean over time → 120-dim vector.

    Output: (N, 120) array per segment
    """

    N_MFCC = 40
    EXPECTED_DIM = MFCC_DIM  # 40 * 3 (static + delta + delta-delta)

    def __init__(self, n_mfcc: int = 40, sr: int = 16000):
        self.n_mfcc = n_mfcc
        self.sr = sr

    # =========================================================
    # UNIFIED: Extract from audio segments (training + inference)
    # =========================================================
    def extract_from_audio(
        self,
        audio_segments: Union[np.ndarray, List[np.ndarray], Sequence[np.ndarray]],
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Extract MFCC + delta + delta-delta from audio segments.

        Accepts both fixed-length chunks (np.ndarray of shape (N, samples))
        and variable-length segments (list of 1-D arrays).

        Args:
            audio_segments: Audio data — either (N, samples_per_chunk) ndarray
                            or list of 1-D arrays with variable lengths.
            sr: Sample rate (default 16000)

        Returns:
            np.ndarray of shape (N, 120)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required. Install with: pip install librosa"
            )

        min_samples = int(_MIN_SEGMENT_SECONDS * sr)

        features_list = []
        for i, chunk in enumerate(audio_segments):
            try:
                chunk = np.asarray(chunk, dtype=np.float32).ravel()
                if chunk.shape[0] < min_samples:
                    chunk = np.pad(chunk, (0, min_samples - chunk.shape[0]))

                # Use n_fft that fits the signal
                n_fft = min(2048, chunk.shape[0])

                mfcc = librosa.feature.mfcc(
                    y=chunk, sr=sr, n_mfcc=self.n_mfcc, n_fft=n_fft,
                )

                # Compute deltas (adapt width to available frames)
                width = min(9, mfcc.shape[1])
                if width < 3:
                    width = 3
                if width % 2 == 0:
                    width -= 1
                delta = librosa.feature.delta(mfcc, width=width)
                delta2 = librosa.feature.delta(mfcc, order=2, width=width)

                stacked = np.vstack([mfcc, delta, delta2])
                feat_vector = stacked.mean(axis=1).astype(np.float32)
                feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)

                if np.allclose(feat_vector, 0.0) or np.isclose(feat_vector.std(), 0.0):
                    logger.warning(
                        f"[VALIDATION_CHECK] Chunk {i}: MFCC vector is near-constant or zero"
                    )

                if feat_vector.shape[0] != self.EXPECTED_DIM:
                    logger.warning(
                        f"[VALIDATION_CHECK] Chunk {i}: expected {self.EXPECTED_DIM}, "
                        f"got {feat_vector.shape[0]}"
                    )
                    if feat_vector.shape[0] > self.EXPECTED_DIM:
                        feat_vector = feat_vector[:self.EXPECTED_DIM]
                    else:
                        pad_width = self.EXPECTED_DIM - feat_vector.shape[0]
                        feat_vector = np.pad(feat_vector, (0, pad_width), constant_values=0.0)

                features_list.append(feat_vector)
            except Exception as e:
                logger.error(f"[DEBUG_POINT] Chunk {i} failed: {e}")
                features_list.append(np.zeros(self.EXPECTED_DIM, dtype=np.float32))

        result = np.stack(features_list, axis=0)
        logger.info(
            f"[DATA_FLOW] MFCC extracted: {result.shape}, "
            f"mean={float(np.mean(result)):.4f}, std={float(np.std(result)):.4f}"
        )
        return result

    # =========================================================
    # UNIFIED INTERFACE
    # =========================================================
    def extract(
        self,
        audio_segments: Union[np.ndarray, List[np.ndarray], None] = None,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Extract MFCCs from audio segments.

        Args:
            audio_segments: Audio data (fixed or variable-length chunks)
            sr: Sample rate
        """
        if audio_segments is not None:
            return self.extract_from_audio(audio_segments, sr)
        else:
            raise ValueError("Must provide audio_segments")
