"""
HuBERT Feature Extractor for Depression Detection.

Replaces MFCC + eGeMAPS with learned speech representations from HuBERT Base.
HuBERT captures prosody, speech rate, pause patterns, vocal quality — all
clinically relevant depression markers — in a single 768-dim vector per segment.

Why HuBERT over MFCC/eGeMAPS:
  - Learned on 960h of speech → captures high-level acoustic patterns
  - Single unified representation vs fragile hand-crafted features
  - Pre-trained representations are robust to recording conditions
  - 768-dim embedding matches text embedding scale (384-dim SBERT)

Uses HuBERT Base (~95M params, ~360MB) which fits in 6GB VRAM.
Extracts frame-level features from the last hidden layer, then mean-pools
over time → one 768-dim vector per segment. Runs in inference mode (frozen).
"""

import logging
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

HUBERT_DIM = 768  # HuBERT Base hidden size
HUBERT_MODEL_ID = "facebook/hubert-base-ls960"
HUBERT_SAMPLE_RATE = 16000
MIN_SAMPLES = 400  # 25ms minimum at 16kHz


class HuBERTExtractor:
    """
    Extract HuBERT Base embeddings from audio segments.

    Produces one 768-dim vector per audio segment by mean-pooling
    the last hidden layer over time frames.

    Thread-safe: model is loaded once and used in eval mode.
    """

    def __init__(
        self,
        model_id: str = HUBERT_MODEL_ID,
        device: Optional[str] = None,
        layer: int = -1,
    ):
        """
        Args:
            model_id: HuggingFace model identifier
            device: "cuda", "cpu", or None (auto-detect)
            layer: Which hidden layer to use (-1 = last)
        """
        self.model_id = model_id
        self.layer = layer
        self._device = device
        self._model = None
        self._processor = None

    def _init_model(self):
        """Lazy-load HuBERT model and processor."""
        if self._model is not None:
            return

        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        logger.info(f"[LAYER_START] Loading HuBERT from {self.model_id}...")
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
        self._model = HubertModel.from_pretrained(self.model_id)
        self._model = self._model.to(device)
        self._model.eval()

        # Freeze all parameters — we only extract features
        for param in self._model.parameters():
            param.requires_grad = False

        logger.info(
            f"[LAYER_START] HuBERT loaded on {device}: "
            f"{sum(p.numel() for p in self._model.parameters())/1e6:.1f}M params (frozen)"
        )

    def extract_single(self, audio: np.ndarray, sr: int = HUBERT_SAMPLE_RATE) -> np.ndarray:
        """
        Extract HuBERT embedding from a single audio segment.

        Args:
            audio: 1-D float32 waveform
            sr: Sample rate (must be 16000)

        Returns:
            np.ndarray of shape (768,) — mean-pooled last hidden state
        """
        self._init_model()

        if sr != HUBERT_SAMPLE_RATE:
            raise ValueError(
                f"HuBERT requires {HUBERT_SAMPLE_RATE}Hz audio, got {sr}Hz"
            )

        # Handle very short segments
        if len(audio) < MIN_SAMPLES:
            # Pad to minimum length
            audio = np.pad(audio, (0, MIN_SAMPLES - len(audio)), mode='constant')

        # Normalize audio to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # Process through HuBERT
        inputs = self._processor(
            audio,
            sampling_rate=HUBERT_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(self._device)

        with torch.no_grad():
            outputs = self._model(input_values, output_hidden_states=True)

        # Get specified layer's hidden states
        if self.layer == -1:
            hidden = outputs.last_hidden_state  # (1, T_frames, 768)
        else:
            hidden = outputs.hidden_states[self.layer]

        # Mean pool over time → (768,)
        embedding = hidden.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
        return embedding

    def extract_from_audio(
        self,
        audio_segments: List[np.ndarray],
        sr: int = HUBERT_SAMPLE_RATE,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Extract HuBERT embeddings from multiple audio segments.

        Args:
            audio_segments: List of 1-D float32 waveforms (variable length)
            sr: Sample rate
            batch_size: Not used for batching (variable lengths), kept for API compat

        Returns:
            np.ndarray of shape (N, 768) where N = len(audio_segments)
        """
        self._init_model()

        if not audio_segments:
            return np.zeros((0, HUBERT_DIM), dtype=np.float32)

        embeddings = []
        for i, segment in enumerate(audio_segments):
            try:
                emb = self.extract_single(segment, sr=sr)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(
                    f"[VALIDATION_CHECK] HuBERT extraction failed for segment {i} "
                    f"(len={len(segment)}): {e}. Using zero vector."
                )
                embeddings.append(np.zeros(HUBERT_DIM, dtype=np.float32))

        result = np.stack(embeddings, axis=0)
        logger.info(f"[DATA_FLOW] HuBERT extracted: {result.shape}")
        return result
