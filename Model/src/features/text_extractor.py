"""
[LAYER_START] Feature Extraction - Text Embedding Extractor
Supports both training (CSV loading) and inference (Whisper Base + SBERT).
Uses Whisper Base (~139MB) instead of WhisperX (~1.5GB) for lower latency.
"""

import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Dict

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Text embedding extractor using Whisper Base + SBERT.

    Training path: Load transcripts from CSV -> encode with SBERT
    Inference path: Transcribe audio chunks with Whisper Base -> encode with SBERT

    Models (lazy-loaded):
        - Whisper Base: openai/whisper-base (~139MB) - inference only
        - SBERT: all-MiniLM-L6-v2 (~80MB, 384-dim) - both paths

    Output: (N, 384) array of sentence embeddings
    """

    EMBEDDING_DIM = 384
    SBERT_MODEL = "all-MiniLM-L6-v2"
    WHISPER_MODEL = "openai/whisper-base"

    def __init__(self, hf_token: Optional[str] = None):
        self._sbert = None
        self._whisper_model = None
        self._whisper_processor = None
        self._device = None

        # Load HF token: explicit arg > env var > .env file
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            env_path = Path(__file__).resolve().parents[2] / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("HF_TOKEN="):
                        self.hf_token = line.split("=", 1)[1].strip()
                        break

    def _get_device(self):
        """Detect best available device."""
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    def _init_sbert(self):
        """Lazy-load SBERT model (needed for both training and inference)."""
        if self._sbert is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sbert = SentenceTransformer(
                    self.SBERT_MODEL,
                    device=self._get_device()
                )
                logger.info(
                    f"[LAYER_START] SBERT '{self.SBERT_MODEL}' loaded on {self._get_device()}"
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

    def _init_whisper(self):
        """Lazy-load Whisper Base model (inference only)."""
        if self._whisper_model is None:
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                import torch

                device = self._get_device()
                self._whisper_processor = WhisperProcessor.from_pretrained(
                    self.WHISPER_MODEL,
                    token=self.hf_token
                )
                self._whisper_model = WhisperForConditionalGeneration.from_pretrained(
                    self.WHISPER_MODEL,
                    token=self.hf_token
                ).to(device)

                logger.info(
                    f"[LAYER_START] Whisper Base loaded on {device}"
                )
            except ImportError:
                raise ImportError(
                    "transformers and torch are required for inference. "
                    "Install with: pip install transformers torch"
                )

    # =========================================================
    # TRAINING PATH: Load transcripts from CSV -> SBERT encode
    # =========================================================
    def load_from_csv(
        self,
        csv_path: Union[str, Path],
        text_column: str = "transcript",
    ) -> np.ndarray:
        """
        [TRAINING_PATH] Load transcripts from CSV and encode with SBERT.

        Handles multiple CSV formats:
        1. Simple: one row per chunk with a transcript column
        2. DAIC-WOZ: start_time|stop_time|speaker|value (tab or comma separated)
           - Filters for participant speech only (excludes interviewer)
           - Groups consecutive participant turns per chunk

        Args:
            csv_path: Path to transcript CSV
            text_column: Column name with transcripts (default: 'transcript')

        Returns:
            np.ndarray of shape (N, 384)
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Transcript CSV not found: {csv_path}")

        # Try comma-separated first, then tab-separated (DAIC-WOZ uses tabs)
        df = pd.read_csv(csv_path, sep=',')
        if len(df.columns) <= 1:
            # Likely tab-separated (DAIC-WOZ format)
            df = pd.read_csv(csv_path, sep='\t')
        logger.info(f"[DATA_FLOW] Loaded transcript CSV: {csv_path.name}, "
                     f"rows={len(df)}, columns={list(df.columns)}")

        # --- Detect DAIC-WOZ format (has speaker + value columns) ---
        speaker_col = None
        for col in df.columns:
            if col.lower() in ('speaker', 'role', 'participant_role'):
                speaker_col = col
                break

        text_col = None
        # Priority: user-specified > 'value' (DAIC-WOZ) > common names
        search_names = [
            text_column.lower(), 'text', 'value', 'transcript',
            'transcription', 'sentence', 'utterance',
        ]
        for target in search_names:
            for col in df.columns:
                if col.lower() == target:
                    text_col = col
                    break
            if text_col is not None:
                break

        if text_col is None:
            raise ValueError(
                f"No transcript column found in {csv_path.name}. "
                f"Available columns: {list(df.columns)}. "
                f"Expected one of: {search_names}"
            )

        # --- Filter for participant speech only (DAIC-WOZ) ---
        if speaker_col is not None:
            participant_aliases = {
                'participant', 'patient', 'subject', 'client', 'interviewee',
            }
            original_count = len(df)
            df = df[
                df[speaker_col].astype(str).str.lower().str.strip().isin(participant_aliases)
            ]
            logger.info(
                f"[DATA_FLOW] Filtered {speaker_col}: {original_count} → {len(df)} "
                f"participant rows (excluded interviewer/Ellie)"
            )

        texts = df[text_col].fillna("").astype(str).tolist()

        if not texts:
            logger.warning(
                f"[VALIDATION_CHECK] No participant text found in {csv_path.name}. "
                f"Returning zero embeddings."
            )
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)

        logger.info(f"[DATA_FLOW] Found {len(texts)} transcripts in column '{text_col}'")

        # Encode with SBERT
        self._init_sbert()
        embeddings = self._sbert.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        ).astype(np.float32)

        logger.info(f"[TRAINING_PATH] Text embeddings: {embeddings.shape}")
        return embeddings

    # =========================================================
    # INFERENCE PATH: Whisper Base transcription + SBERT encode
    # =========================================================
    def transcribe_chunks(
        self,
        audio_chunks: np.ndarray,
        sr: int = 16000,
        chunk_start_times: Optional[List[float]] = None,
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        [INFERENCE_PATH] Transcribe audio chunks using Whisper Base (batched).

        Processes all chunks in batches instead of one-by-one to minimize
        model overhead and GPU round-trips (major latency win).

        Args:
            audio_chunks: np.ndarray of shape (num_chunks, samples)
            sr: Sample rate (default 16000)
            chunk_start_times: Start time of each chunk for global timestamps
            batch_size: Number of chunks to process per Whisper forward pass

        Returns:
            List of dicts with keys: 'text', 'chunk_idx', 'start_time', 'end_time'
        """
        self._init_whisper()
        import torch

        device = self._get_device()
        num_chunks = len(audio_chunks)
        all_texts: List[str] = []

        # Process in batches — one Whisper forward pass per batch
        for batch_start in range(0, num_chunks, batch_size):
            batch = audio_chunks[batch_start: batch_start + batch_size]
            try:
                inputs = self._whisper_processor(
                    list(batch), sampling_rate=sr, return_tensors="pt", padding=True
                )
                input_features = inputs.input_features.to(device)
                with torch.no_grad():
                    predicted_ids = self._whisper_model.generate(input_features)
                texts = self._whisper_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )
                all_texts.extend([t.strip() for t in texts])
            except Exception as e:
                logger.error(
                    f"[DEBUG_POINT] Batch [{batch_start}:{batch_start+len(batch)}] "
                    f"transcription failed: {e}"
                )
                all_texts.extend([""] * len(batch))

        results = []
        for i, text in enumerate(all_texts):
            chunk_len = len(audio_chunks[i])
            chunk_start = chunk_start_times[i] if chunk_start_times else i * (chunk_len / sr)
            chunk_end = chunk_start + chunk_len / sr
            results.append({
                'text': text,
                'chunk_idx': i,
                'start_time': chunk_start,
                'end_time': chunk_end,
            })

        non_empty = sum(1 for r in results if r['text'])
        logger.info(
            f"[INFERENCE_PATH] Transcribed {non_empty}/{len(results)} chunks "
            f"(batch_size={batch_size})"
        )
        return results

    def extract_from_audio(
        self,
        audio_chunks: np.ndarray,
        sr: int = 16000,
        chunk_start_times: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        [INFERENCE_PATH] Full pipeline: transcribe + embed audio chunks.

        Returns:
            np.ndarray of shape (num_chunks, 384)
        """
        # Step 1: Transcribe
        transcriptions = self.transcribe_chunks(audio_chunks, sr, chunk_start_times)

        # Step 2: Encode with SBERT
        texts = [t['text'] for t in transcriptions]
        self._init_sbert()
        embeddings = self._sbert.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        logger.info(f"[INFERENCE_PATH] Text embeddings: {embeddings.shape}")
        return embeddings

    # =========================================================
    # UNIFIED INTERFACE
    # =========================================================
    def extract(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        audio_chunks: Optional[np.ndarray] = None,
        sr: int = 16000,
        text_column: str = "transcript",
        chunk_start_times: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Unified extraction interface.

        - If csv_path is provided -> training path (CSV -> SBERT encode)
        - If audio_chunks is provided -> inference path (Whisper -> SBERT encode)
        """
        if csv_path is not None:
            return self.load_from_csv(csv_path, text_column)
        elif audio_chunks is not None:
            return self.extract_from_audio(audio_chunks, sr, chunk_start_times)
        else:
            raise ValueError(
                "Must provide either csv_path (training) or audio_chunks (inference)"
            )
