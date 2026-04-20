"""
=============================================================================
[LAYER_START] Session 2: Audio Chunking
=============================================================================

Splits audio signals into fixed-duration overlapping chunks.
- Configurable chunk duration and overlap
- Discards chunks below minimum duration
- Optional max chunks per subject cap
- Returns chunk arrays + metadata (timestamps)

[TRAINING_PATH] Chunk all preprocessed training audio
[INFERENCE_PATH] Chunk single preprocessed audio with identical logic
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# [DATA_FLOW] Chunk representation
@dataclass
class AudioChunk:
    """Represents a single audio chunk with metadata."""
    chunk_index: int
    audio: np.ndarray
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    duration: float


@dataclass
class ChunkResult:
    """Result of chunking a single audio file."""
    participant_id: str
    chunks: List[AudioChunk] = field(default_factory=list)
    total_audio_duration: float = 0.0
    num_chunks: int = 0
    sample_rate: int = 16000


class AudioChunker:
    """
    Splits audio into fixed-duration overlapping chunks.

    Parameters
    ----------
    chunk_duration : float
        Duration of each chunk in seconds (default: 5.0).
    overlap : float
        Fraction of overlap between consecutive chunks (0.0 to 0.5).
    min_chunk_duration : float
        Minimum acceptable chunk duration in seconds. Chunks shorter
        than this (typically the last one) are discarded.
    max_chunks_per_subject : int
        Maximum number of chunks to keep per subject. 0 = no limit.
    sample_rate : int
        Audio sample rate.
    """

    def __init__(
        self,
        chunk_duration: float = 5.0,
        overlap: float = 0.25,
        min_chunk_duration: float = 2.0,
        max_chunks_per_subject: int = 0,
        sample_rate: int = 16000,
    ):
        # [VALIDATION_CHECK] Parameter bounds
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"Overlap must be in [0, 1), got {overlap}")
        if chunk_duration <= 0:
            raise ValueError(f"chunk_duration must be > 0, got {chunk_duration}")
        if min_chunk_duration < 0:
            raise ValueError(
                f"min_chunk_duration must be >= 0, got {min_chunk_duration}"
            )

        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.min_chunk_duration = min_chunk_duration
        self.max_chunks_per_subject = max_chunks_per_subject
        self.sample_rate = sample_rate

        # Derived parameters
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.hop_samples = int(self.chunk_samples * (1 - overlap))
        self.min_chunk_samples = int(min_chunk_duration * sample_rate)

    def chunk(
        self,
        audio: np.ndarray,
        participant_id: str = "unknown",
    ) -> ChunkResult:
        """
        Split audio into overlapping chunks.

        Parameters
        ----------
        audio : np.ndarray
            1D float audio signal (mono).
        participant_id : str
            Identifier for the subject.

        Returns
        -------
        ChunkResult
            Contains list of AudioChunk objects and metadata.

        [DATA_FLOW] audio (1D) → List[AudioChunk]
        [DEBUG_POINT] Returns ChunkResult with full metadata for inspection.
        """
        # [VALIDATION_CHECK] Input validation
        if audio.ndim != 1:
            raise ValueError(
                f"Expected 1D audio array, got shape {audio.shape}"
            )

        result = ChunkResult(
            participant_id=participant_id,
            total_audio_duration=len(audio) / self.sample_rate,
            sample_rate=self.sample_rate,
        )

        if len(audio) == 0:
            return result

        # [DATA_FLOW] Sliding window chunking
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(audio):
            end = min(start + self.chunk_samples, len(audio))
            chunk_audio = audio[start:end]
            chunk_len = len(chunk_audio)

            # Discard chunks shorter than minimum
            if chunk_len < self.min_chunk_samples:
                break

            # Pad short final chunk if it meets minimum threshold
            if chunk_len < self.chunk_samples:
                padded = np.zeros(self.chunk_samples, dtype=audio.dtype)
                padded[:chunk_len] = chunk_audio
                chunk_audio = padded

            chunk = AudioChunk(
                chunk_index=chunk_idx,
                audio=chunk_audio,
                start_sample=start,
                end_sample=end,
                start_time=start / self.sample_rate,
                end_time=end / self.sample_rate,
                duration=chunk_len / self.sample_rate,
            )
            chunks.append(chunk)
            chunk_idx += 1

            start += self.hop_samples

        # [VALIDATION_CHECK] Apply max chunks cap
        if self.max_chunks_per_subject > 0 and len(chunks) > self.max_chunks_per_subject:
            chunks = chunks[: self.max_chunks_per_subject]

        result.chunks = chunks
        result.num_chunks = len(chunks)

        return result

    def get_chunk_arrays(
        self, chunk_result: ChunkResult
    ) -> Optional[np.ndarray]:
        """
        Stack all chunk audio arrays into a 2D numpy array.

        Returns
        -------
        np.ndarray or None
            Shape: (num_chunks, chunk_samples). None if no chunks.

        [DATA_FLOW] ChunkResult → stacked array for downstream feature extraction
        """
        if not chunk_result.chunks:
            return None
        return np.stack([c.audio for c in chunk_result.chunks], axis=0)

    def get_chunk_metadata(self, chunk_result: ChunkResult) -> List[dict]:
        """
        Extract metadata (timestamps, indices) for all chunks.

        [DEBUG_POINT] Useful for aligning features across modalities.
        """
        return [
            {
                "participant_id": chunk_result.participant_id,
                "chunk_index": c.chunk_index,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "duration": c.duration,
            }
            for c in chunk_result.chunks
        ]
