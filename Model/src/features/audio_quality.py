"""
Audio Quality Scorer — measures per-segment audio usefulness.

Computes three quality metrics:
  1. SNR (Signal-to-Noise Ratio) — estimated via power ratio
  2. RMS Energy — raw signal energy
  3. Speech Probability — fraction of frames above energy threshold

Outputs a scalar quality score [0, 1] per segment.
Used by gated fusion to dynamically weight audio contribution.
Segments with low quality → reduced audio weight → text dominates.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioQualityScorer:
    """
    Computes per-segment audio quality scores for dynamic fusion weighting.

    Quality score = weighted combination of:
      - SNR (higher = cleaner audio)
      - RMS energy (too low = near-silent)
      - Speech probability (higher = more speech content)

    Output: float in [0, 1] per segment.
    """

    def __init__(
        self,
        frame_length: float = 0.025,
        hop_length: float = 0.010,
        energy_floor: float = 1e-10,
        silence_threshold_db: float = -40.0,
        snr_min: float = 0.0,
        snr_max: float = 40.0,
        energy_weight: float = 0.3,
        snr_weight: float = 0.4,
        speech_prob_weight: float = 0.3,
        sample_rate: int = 16000,
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_floor = energy_floor
        self.silence_threshold_db = silence_threshold_db
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.energy_weight = energy_weight
        self.snr_weight = snr_weight
        self.speech_prob_weight = speech_prob_weight
        self.sample_rate = sample_rate

    def compute_rms(self, audio: np.ndarray) -> float:
        """Root mean square energy."""
        return float(np.sqrt(np.mean(audio ** 2) + self.energy_floor))

    def estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate SNR by comparing top-energy frames (signal) vs bottom frames (noise).

        Simple but effective: sort frame energies, use top 80% as signal, bottom 20% as noise.
        """
        frame_samples = int(self.frame_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)

        if len(audio) < frame_samples:
            return 0.0

        n_frames = max(1, 1 + (len(audio) - frame_samples) // hop_samples)
        energies = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_samples
            end = min(start + frame_samples, len(audio))
            frame = audio[start:end]
            energies[i] = np.mean(frame ** 2) + self.energy_floor

        sorted_e = np.sort(energies)
        n_noise = max(1, int(0.2 * n_frames))
        noise_power = np.mean(sorted_e[:n_noise])
        signal_power = np.mean(sorted_e[n_noise:]) if n_frames > n_noise else noise_power

        if noise_power <= self.energy_floor:
            return self.snr_max

        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(np.clip(snr_db, self.snr_min, self.snr_max))

    def compute_speech_probability(self, audio: np.ndarray) -> float:
        """Fraction of frames with energy above silence threshold."""
        frame_samples = int(self.frame_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)

        if len(audio) < frame_samples:
            return 0.0

        threshold_linear = 10 ** (self.silence_threshold_db / 20)

        n_frames = max(1, 1 + (len(audio) - frame_samples) // hop_samples)
        speech_frames = 0
        for i in range(n_frames):
            start = i * hop_samples
            end = min(start + frame_samples, len(audio))
            frame = audio[start:end]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > threshold_linear:
                speech_frames += 1

        return speech_frames / n_frames

    def score_segment(self, audio: np.ndarray) -> Tuple[float, dict]:
        """
        Compute overall quality score for one audio segment.

        Args:
            audio: 1-D float32 waveform

        Returns:
            (quality_score, details_dict)
            quality_score: float in [0, 1]
            details_dict: individual metric values
        """
        rms = self.compute_rms(audio)
        snr = self.estimate_snr(audio)
        speech_prob = self.compute_speech_probability(audio)

        # Normalize SNR to [0, 1]
        snr_norm = (snr - self.snr_min) / (self.snr_max - self.snr_min)
        snr_norm = float(np.clip(snr_norm, 0, 1))

        # Normalize RMS energy to [0, 1] via sigmoid-like curve
        # RMS of 0.01 → ~0.5, RMS of 0.1 → ~0.95
        rms_norm = float(np.clip(1 - np.exp(-rms * 100), 0, 1))

        quality = (
            self.energy_weight * rms_norm
            + self.snr_weight * snr_norm
            + self.speech_prob_weight * speech_prob
        )
        quality = float(np.clip(quality, 0, 1))

        details = {
            'rms': rms,
            'rms_norm': rms_norm,
            'snr_db': snr,
            'snr_norm': snr_norm,
            'speech_prob': speech_prob,
            'quality': quality,
        }
        return quality, details

    def score_segments(self, audio_segments: List[np.ndarray]) -> np.ndarray:
        """
        Score multiple audio segments.

        Args:
            audio_segments: List of 1-D float32 waveforms

        Returns:
            np.ndarray of shape (N,) with quality scores in [0, 1]
        """
        if not audio_segments:
            return np.array([], dtype=np.float32)

        scores = []
        for seg in audio_segments:
            score, _ = self.score_segment(seg)
            scores.append(score)

        result = np.array(scores, dtype=np.float32)
        logger.info(
            f"[DATA_FLOW] Audio quality scored: N={len(result)}, "
            f"mean={result.mean():.3f}, min={result.min():.3f}, max={result.max():.3f}"
        )
        return result
