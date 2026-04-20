"""
Fusion Model Predictor — DepressoSpeech

Loads the trained MultimodalFusion checkpoint and runs deterministic inference.
Handles text(384) + mfcc(120) + egemaps(88) + behavioral(16) features.
"""

import numpy as np
import hashlib
import torch
import logging
from pathlib import Path
from typing import Optional, Union

from src.models.multimodal_fusion import MultimodalFusion

logger = logging.getLogger(__name__)


class FusionPredictor:
    """
    Wraps MultimodalFusion for deterministic inference.

    Prediction flow:
        1. Load pretrained text backbone + fusion checkpoint
        2. Receive per-segment text, mfcc, egemaps + interview-level behavioral features
        3. StatsPool → BN → text_pred + residual → PHQ-8 score
    """

    def __init__(
        self,
        fusion_checkpoint: Union[str, Path],
        text_checkpoint: Union[str, Path] = "checkpoints/best_model.pt",
        device: str = "auto",
    ):
        self.device = self._resolve_device(device)

        # Build model
        self.model = MultimodalFusion(stats_mode="mean_std").to(self.device)
        self.model.load_pretrained_text(str(text_checkpoint))

        # Load fusion weights
        ckpt = torch.load(
            str(fusion_checkpoint), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.metadata = {
            "epoch": ckpt.get("epoch", -1),
            "val_ccc": ckpt.get("val_ccc", None),
            "model_version": self._compute_version(ckpt),
        }

        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"FusionPredictor loaded: epoch={self.metadata['epoch']}, "
            f"val_ccc={self.metadata['val_ccc']:.4f}, params={total:,}, "
            f"device={self.device}"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _compute_version(ckpt: dict) -> str:
        state = ckpt.get("model_state_dict", {})
        h = hashlib.sha256()
        for key in sorted(state.keys()):
            h.update(key.encode())
            h.update(state[key].cpu().numpy().tobytes())
        return h.hexdigest()[:12]

    @torch.no_grad()
    def predict(
        self,
        text_features: np.ndarray,
        mfcc_features: np.ndarray,
        egemaps_features: np.ndarray,
        behavioral: Optional[np.ndarray] = None,
        normalize_text: bool = True,
        text_scale: float = 0.6,
    ) -> float:
        """
        Predict PHQ-8 score from aligned per-segment features.

        Args:
            text_features:    (N, 384) per-segment text embeddings
            mfcc_features:    (N, 120) per-segment MFCC features
            egemaps_features: (N, 88)  per-segment eGeMAPS features
            behavioral:       (16,) interview-level behavioral features, or None
            normalize_text:   Whether to L2-normalize text. Set False if already normalized.
            text_scale:       Weight for text branch contribution [0,1]. Default 0.6
                              reduces text dominance so audio features contribute more.

        Returns:
            PHQ-8 score clamped to [0, 24]
        """
        if normalize_text:
            norms = np.linalg.norm(text_features, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            text_features = text_features / norms

        # Per-session z-score normalization of audio features.
        # Removes session-level mean/scale differences so the model sees
        # feature PATTERNS (intonation, rhythm) rather than absolute values.
        mfcc_features = self._session_normalize(mfcc_features)
        egemaps_features = self._session_normalize(egemaps_features)

        # Ensure float32
        text_t = torch.from_numpy(text_features.astype(np.float32)).unsqueeze(0).to(self.device)
        mfcc_t = torch.from_numpy(mfcc_features.astype(np.float32)).unsqueeze(0).to(self.device)
        egemaps_t = torch.from_numpy(egemaps_features.astype(np.float32)).unsqueeze(0).to(self.device)

        N = text_features.shape[0]
        mask = torch.ones(1, N, dtype=torch.bool, device=self.device)

        if behavioral is not None:
            behavioral_t = torch.from_numpy(behavioral.astype(np.float32)).unsqueeze(0).to(self.device)
        else:
            behavioral_t = torch.zeros(1, 16, device=self.device)

        pred = self.model(text_t, mfcc_t, egemaps_t, mask, behavioral_t, text_scale=text_scale)
        return float(pred.squeeze().clamp(0.0, 24.0).item())

    @staticmethod
    def _session_normalize(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Per-session (within-recording) z-score normalization.

        Subtracts the session mean and divides by session std so the model
        receives relative feature patterns rather than absolute speaker-level
        values. This prevents a calm-voiced speaker from always scoring low
        and a loud speaker from always scoring high.

        Args:
            features: (N, D) array of raw per-chunk features
            eps: small constant to avoid division by zero

        Returns:
            (N, D) session-normalized features
        """
        if features.shape[0] < 2:
            return features
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return ((features - mean) / std).astype(np.float32)
