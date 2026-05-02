"""
=============================================================================
[LAYER_START] Session 2: Audio Preprocessor (Orchestrator)
=============================================================================

End-to-end audio preprocessing pipeline:
1. Load audio file
2. Resample to target sample rate (16kHz)
3. Convert to mono
4. Apply Voice Activity Detection (VAD)
5. Chunk into fixed-duration overlapping windows
6. Save processed chunks + metadata

[TRAINING_PATH] Batch preprocessing of all training/dev/test audio
[INFERENCE_PATH] Single-file preprocessing with identical pipeline
=============================================================================
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm

from src.preprocessing.vad import EnergyVAD
from src.preprocessing.chunker import AudioChunker, ChunkResult

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Orchestrates the full audio preprocessing pipeline.

    Pipeline: Load → Resample → Mono → VAD → Chunk → Save

    Parameters
    ----------
    config_path : str or None
        Path to audio_config.yaml. If None, uses default parameters.
    config : dict or None
        Direct config dict (overrides config_path).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self._init_components()

    @staticmethod
    def _default_config() -> dict:
        return {
            "audio": {"sample_rate": 16000},
            "preprocessing": {
                "vad": {
                    "backend": "energy",
                    "energy_threshold": 0.02,
                    "frame_length": 0.03,
                    "hop_length": 0.01,
                    "min_speech_duration": 0.3,
                    "max_silence_duration": 0.5,
                },
                "chunking": {
                    "chunk_duration": 5.0,
                    "overlap": 0.25,
                    "min_chunk_duration": 2.0,
                    "max_chunks_per_subject": 0,
                },
            },
            "paths": {
                "raw_audio_dir": "data/raw",
                "processed_audio_dir": "data/processed/audio_chunks",
                "chunk_metadata_dir": "data/processed/chunk_metadata",
            },
        }

    def _init_components(self):
        """Initialize VAD and Chunker from config."""
        audio_cfg = self.config["audio"]
        vad_cfg = self.config["preprocessing"]["vad"]
        chunk_cfg = self.config["preprocessing"]["chunking"]

        self.sample_rate = audio_cfg["sample_rate"]

        # [LAYER_START] VAD component
        self.vad = EnergyVAD(
            energy_threshold=vad_cfg["energy_threshold"],
            frame_length=vad_cfg["frame_length"],
            hop_length=vad_cfg["hop_length"],
            min_speech_duration=vad_cfg["min_speech_duration"],
            max_silence_duration=vad_cfg["max_silence_duration"],
            sample_rate=self.sample_rate,
        )

        # [LAYER_START] Chunker component
        self.chunker = AudioChunker(
            chunk_duration=chunk_cfg["chunk_duration"],
            overlap=chunk_cfg["overlap"],
            min_chunk_duration=chunk_cfg["min_chunk_duration"],
            max_chunks_per_subject=chunk_cfg.get("max_chunks_per_subject", 0),
            sample_rate=self.sample_rate,
        )

    # =========================================================================
    # Core Processing
    # =========================================================================

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file, resample to target rate, convert to mono.

        [DATA_FLOW] file path → (audio_array, sample_rate)
        [VALIDATION_CHECK] Verifies file exists and is readable.
        """
        audio_path = str(audio_path)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load and resample
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # [VALIDATION_CHECK] Ensure float32
        audio = audio.astype(np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak

        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        min_amp = float(audio.min()) if audio.size else 0.0
        max_amp = float(audio.max()) if audio.size else 0.0

        # [DEBUG_POINT] Log audio stats
        logger.info(
            f"Loaded {audio_path}: duration={len(audio)/sr:.2f}s, "
            f"sr={sr}, waveform_len={len(audio)}, min={min_amp:.4f}, "
            f"max={max_amp:.4f}, rms={rms:.4f}"
        )

        return audio, sr

    def process_single(
        self,
        audio_path: str,
        participant_id: str = "unknown",
    ) -> ChunkResult:
        """
        Process a single audio file through the full pipeline.

        [TRAINING_PATH] / [INFERENCE_PATH] — identical logic for both.

        Pipeline:
        1. Load + resample + mono
        2. Apply VAD
        3. Chunk into windows

        Parameters
        ----------
        audio_path : str
            Path to the audio file.
        participant_id : str
            Subject identifier.

        Returns
        -------
        ChunkResult
            Processed chunks with metadata.
        """
        # Step 1: Load audio
        # [DATA_FLOW] file → raw audio
        audio, sr = self.load_audio(audio_path)
        original_duration = len(audio) / sr

        # [DEBUG_POINT] Check for empty/corrupt audio
        if len(audio) == 0:
            logger.warning(f"Empty audio for participant {participant_id}")
            return ChunkResult(
                participant_id=participant_id,
                total_audio_duration=0.0,
                sample_rate=sr,
            )

        # Step 2: Apply VAD
        # [DATA_FLOW] raw audio → VAD-filtered audio
        vad_audio, segments = self.vad.detect(audio)
        vad_duration = len(vad_audio) / sr

        logger.info(
            f"[{participant_id}] VAD: {original_duration:.1f}s → "
            f"{vad_duration:.1f}s ({len(segments)} segments)"
        )

        # [DEBUG_POINT] Check VAD output
        if len(vad_audio) == 0:
            logger.warning(
                f"No speech detected for participant {participant_id}"
            )
            return ChunkResult(
                participant_id=participant_id,
                total_audio_duration=original_duration,
                raw_audio_duration=original_duration,
                vad_audio_duration=0.0,
                silence_duration=original_duration,
                sample_rate=sr,
            )

        # Step 3: Chunk
        # [DATA_FLOW] VAD audio → chunks
        chunk_result = self.chunker.chunk(vad_audio, participant_id)
        chunk_result.total_audio_duration = original_duration
        chunk_result.raw_audio_duration = original_duration
        chunk_result.vad_audio_duration = vad_duration
        chunk_result.silence_duration = max(original_duration - vad_duration, 0.0)

        logger.info(
            f"[{participant_id}] Chunking: {chunk_result.num_chunks} chunks "
            f"({self.chunker.chunk_duration}s each, "
            f"{self.chunker.overlap*100:.0f}% overlap)"
        )

        return chunk_result

    # =========================================================================
    # [TRAINING_PATH] Batch Processing
    # =========================================================================

    def process_batch(
        self,
        audio_dir: str,
        participant_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        save_chunks: bool = True,
    ) -> Dict[str, ChunkResult]:
        """
        Batch process multiple audio files.

        Parameters
        ----------
        audio_dir : str
            Directory containing participant audio folders/files.
        participant_ids : list of str or None
            Specific participants to process. If None, process all found.
        output_dir : str or None
            Directory to save processed chunks. If None, uses config path.
        save_chunks : bool
            Whether to save chunk audio files to disk.

        Returns
        -------
        dict
            Mapping of participant_id → ChunkResult.

        [TRAINING_PATH] Process all training audio in batch.
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        if output_dir is None:
            output_dir = self.config["paths"]["processed_audio_dir"]
        output_dir = Path(output_dir)

        # Discover audio files
        audio_files = self._discover_audio_files(audio_dir, participant_ids)

        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return {}

        logger.info(f"Processing {len(audio_files)} audio files from {audio_dir}")

        results = {}
        for pid, fpath in tqdm(audio_files.items(), desc="Preprocessing audio"):
            try:
                chunk_result = self.process_single(str(fpath), pid)
                results[pid] = chunk_result

                if save_chunks and chunk_result.num_chunks > 0:
                    self._save_chunks(chunk_result, output_dir)

            except Exception as e:
                # [DEBUG_POINT] Log and skip failed files
                logger.error(f"Failed to process {pid}: {e}")
                continue

        # Save batch metadata
        if save_chunks:
            self._save_batch_metadata(results, output_dir)

        logger.info(
            f"Batch complete: {len(results)}/{len(audio_files)} files processed"
        )

        return results

    # =========================================================================
    # [INFERENCE_PATH] Single-file inference preprocessing
    # =========================================================================

    def preprocess_for_inference(
        self, audio_path: str, participant_id: str = "inference"
    ) -> Optional[np.ndarray]:
        """
        Preprocess a single audio file for inference.

        Returns the stacked chunk array ready for feature extraction.

        [INFERENCE_PATH] Exact same pipeline as training.

        Returns
        -------
        np.ndarray or None
            Shape: (num_chunks, chunk_samples). None if no valid chunks.
        """
        chunk_result = self.process_single(audio_path, participant_id)

        if chunk_result.num_chunks == 0:
            logger.warning(
                f"No valid chunks produced for inference: {audio_path}"
            )
            return None

        return self.chunker.get_chunk_arrays(chunk_result)

    # =========================================================================
    # I/O Helpers
    # =========================================================================

    def _discover_audio_files(
        self,
        audio_dir: Path,
        participant_ids: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Discover audio files in directory.

        Supports two layouts:
        1. Flat: audio_dir/<participant_id>.wav
        2. Nested: audio_dir/<participant_id>/<participant_id>.wav

        [DEBUG_POINT] Logs discovery stats.
        """
        supported = {".wav", ".mp3", ".flac"}
        audio_files = {}

        for entry in sorted(audio_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in supported:
                pid = entry.stem
                if participant_ids is None or pid in participant_ids:
                    audio_files[pid] = entry
            elif entry.is_dir():
                pid = entry.name
                if participant_ids is not None and pid not in participant_ids:
                    continue
                for f in sorted(entry.iterdir()):
                    if f.is_file() and f.suffix.lower() in supported:
                        audio_files[pid] = f
                        break

        logger.debug(f"Discovered {len(audio_files)} audio files")
        return audio_files

    def _save_chunks(self, chunk_result: ChunkResult, output_dir: Path):
        """
        Save individual chunk audio files.

        [DATA_FLOW] ChunkResult → disk (wav files + metadata JSON)
        """
        pid_dir = output_dir / chunk_result.participant_id
        pid_dir.mkdir(parents=True, exist_ok=True)

        for chunk in chunk_result.chunks:
            chunk_path = pid_dir / f"chunk_{chunk.chunk_index:04d}.wav"
            sf.write(str(chunk_path), chunk.audio, chunk_result.sample_rate)

        # Save per-participant metadata
        metadata = self.chunker.get_chunk_metadata(chunk_result)
        meta_path = pid_dir / "chunk_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_batch_metadata(
        self, results: Dict[str, ChunkResult], output_dir: Path
    ):
        """
        Save summary metadata for entire batch.

        [DEBUG_POINT] Useful for verifying batch processing results.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {}
        for pid, cr in results.items():
            summary[pid] = {
                "num_chunks": cr.num_chunks,
                "total_audio_duration": cr.total_audio_duration,
                "sample_rate": cr.sample_rate,
            }

        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Batch metadata saved to {summary_path}")
