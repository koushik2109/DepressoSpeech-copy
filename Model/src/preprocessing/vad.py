"""
=============================================================================
[LAYER_START] Session 2: Voice Activity Detection (VAD)
=============================================================================

Detects speech vs. silence regions in audio signals.
- Removes long silence segments
- Preserves short pauses (clinically relevant for depression detection)
- Supports energy-based VAD (default) with extensible backend design

[TRAINING_PATH] Batch VAD across all training audio files
[INFERENCE_PATH] Single-file VAD with identical logic
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


# [DATA_FLOW] VAD segment representation
@dataclass
class SpeechSegment:
    """Represents a detected speech segment."""
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    is_speech: bool


class EnergyVAD:
    """
    Energy-based Voice Activity Detection.

    Classifies audio frames as speech or silence based on short-term energy.
    Preserves short pauses (important for depression speech patterns).

    Parameters
    ----------
    energy_threshold : float
        Fraction of max energy below which a frame is considered silence.
    frame_length : float
        Duration of each analysis frame in seconds.
    hop_length : float
        Hop between consecutive frames in seconds.
    min_speech_duration : float
        Minimum speech segment duration to keep (seconds).
    max_silence_duration : float
        Maximum silence duration to preserve (seconds). Silence longer
        than this is removed; shorter silence is kept as pauses.
    sample_rate : int
        Audio sample rate.
    """

    def __init__(
        self,
        energy_threshold: float = 0.02,
        frame_length: float = 0.03,
        hop_length: float = 0.01,
        min_speech_duration: float = 0.3,
        max_silence_duration: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.energy_threshold = energy_threshold
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.min_speech_duration = min_speech_duration
        self.max_silence_duration = max_silence_duration
        self.sample_rate = sample_rate

        # Derived parameters
        self.frame_samples = int(frame_length * sample_rate)
        self.hop_samples = int(hop_length * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_silence_samples = int(max_silence_duration * sample_rate)

    def _compute_frame_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute short-term energy per frame.

        [DATA_FLOW] audio (1D float) → frame energies (1D float)
        """
        num_frames = max(
            1, 1 + (len(audio) - self.frame_samples) // self.hop_samples
        )
        energies = np.zeros(num_frames, dtype=np.float64)

        for i in range(num_frames):
            start = i * self.hop_samples
            end = min(start + self.frame_samples, len(audio))
            frame = audio[start:end]
            energies[i] = np.sum(frame ** 2) / len(frame)

        return energies

    def _classify_frames(self, energies: np.ndarray) -> np.ndarray:
        """
        Classify frames as speech (1) or silence (0).

        [VALIDATION_CHECK] Threshold is relative to max energy in the signal.
        """
        max_energy = np.max(energies)
        if max_energy == 0:
            # [DEBUG_POINT] Entire signal is silence
            return np.zeros(len(energies), dtype=np.int32)

        threshold = self.energy_threshold * max_energy
        return (energies > threshold).astype(np.int32)

    def _frames_to_segments(
        self, frame_labels: np.ndarray, audio_num_samples: int
    ) -> List[SpeechSegment]:
        """
        Convert frame-level labels to time-aligned segments.

        [DATA_FLOW] frame_labels → List[SpeechSegment]
        """
        segments = []
        if len(frame_labels) == 0:
            return segments

        current_label = frame_labels[0]
        start_frame = 0

        for i in range(1, len(frame_labels)):
            if frame_labels[i] != current_label:
                start_sample = start_frame * self.hop_samples
                end_sample = i * self.hop_samples
                segments.append(
                    SpeechSegment(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        start_time=start_sample / self.sample_rate,
                        end_time=end_sample / self.sample_rate,
                        is_speech=bool(current_label),
                    )
                )
                current_label = frame_labels[i]
                start_frame = i

        # Final segment
        start_sample = start_frame * self.hop_samples
        end_sample = audio_num_samples
        segments.append(
            SpeechSegment(
                start_sample=start_sample,
                end_sample=end_sample,
                start_time=start_sample / self.sample_rate,
                end_time=end_sample / self.sample_rate,
                is_speech=bool(current_label),
            )
        )

        return segments

    def _filter_segments(
        self, segments: List[SpeechSegment]
    ) -> List[SpeechSegment]:
        """
        Filter segments: remove short speech, preserve short pauses.

        Rules:
        - Speech segments shorter than min_speech_duration → removed
        - Silence segments shorter than max_silence_duration → kept (pauses)
        - Silence segments longer than max_silence_duration → removed

        [VALIDATION_CHECK] Ensures clinically relevant pauses are preserved.
        """
        filtered = []
        for seg in segments:
            duration_samples = seg.end_sample - seg.start_sample
            if seg.is_speech:
                if duration_samples >= self.min_speech_samples:
                    filtered.append(seg)
            else:
                # Keep short pauses, remove long silence
                if duration_samples <= self.max_silence_samples:
                    filtered.append(seg)
        return filtered

    def detect(self, audio: np.ndarray) -> Tuple[np.ndarray, List[SpeechSegment]]:
        """
        Run VAD on audio signal.

        Parameters
        ----------
        audio : np.ndarray
            1D float audio signal (mono, at self.sample_rate).

        Returns
        -------
        vad_audio : np.ndarray
            Audio with long silence removed, short pauses preserved.
        segments : List[SpeechSegment]
            Detected and filtered speech/pause segments.

        [DEBUG_POINT] Returns both audio and segments for inspection.
        """
        # [VALIDATION_CHECK] Input validation
        if audio.ndim != 1:
            raise ValueError(
                f"Expected 1D audio array, got shape {audio.shape}"
            )
        if len(audio) == 0:
            return np.array([], dtype=audio.dtype), []

        # Step 1: Compute frame energies
        energies = self._compute_frame_energy(audio)

        # Step 2: Classify frames
        frame_labels = self._classify_frames(energies)

        # Step 3: Convert to segments
        segments = self._frames_to_segments(frame_labels, len(audio))

        # Step 4: Filter segments
        filtered_segments = self._filter_segments(segments)

        # Step 5: Reconstruct audio from filtered segments
        if not filtered_segments:
            # [DEBUG_POINT] No speech detected
            return np.array([], dtype=audio.dtype), []

        audio_parts = []
        for seg in filtered_segments:
            end = min(seg.end_sample, len(audio))
            audio_parts.append(audio[seg.start_sample:end])

        vad_audio = np.concatenate(audio_parts)

        return vad_audio, filtered_segments


def apply_vad(
    audio: np.ndarray,
    sample_rate: int = 16000,
    energy_threshold: float = 0.02,
    min_speech_duration: float = 0.3,
    max_silence_duration: float = 0.5,
    frame_length: float = 0.03,
    hop_length: float = 0.01,
) -> Tuple[np.ndarray, List[SpeechSegment]]:
    """
    Convenience function to apply VAD with default parameters.

    [TRAINING_PATH] / [INFERENCE_PATH] — same function for both paths.
    """
    vad = EnergyVAD(
        energy_threshold=energy_threshold,
        frame_length=frame_length,
        hop_length=hop_length,
        min_speech_duration=min_speech_duration,
        max_silence_duration=max_silence_duration,
        sample_rate=sample_rate,
    )
    return vad.detect(audio)
