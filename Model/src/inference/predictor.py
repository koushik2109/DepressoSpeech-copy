"""
[LAYER_START] Session 8: Model Predictor
Loads trained DepressionModel checkpoint and runs deterministic inference.

[INFERENCE_PATH] Load saved weights → eval mode → predict PHQ-8 scores.
"""

import numpy as np
import hashlib
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Union

from src.models import DepressionModel

logger = logging.getLogger(__name__)


class Predictor:
    """
    Wraps DepressionModel for deterministic inference.

    Handles:
        - Loading trained checkpoint (model weights + metadata)
        - Device management (auto-detect CPU/CUDA)
        - eval() mode enforcement (no dropout, no gradients)
        - PHQ-8 score prediction from PCA-reduced features
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_config: Optional[dict] = None,
        device: str = "auto",
    ):
        """
        Args:
            model_path: Path to saved checkpoint (.pt file from Trainer)
            model_config: Model architecture config. If None, loaded from checkpoint.
            device: "auto", "cpu", or "cuda"
        """
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)

        # Load checkpoint
        checkpoint = self._load_checkpoint(self.model_path)

        # Build model from config
        cfg = model_config or checkpoint.get('model_config', {})
        self.model = self._build_model(cfg)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Store metadata
        self.metadata = {
            'epoch': checkpoint.get('epoch', -1),
            'best_metric': checkpoint.get('best_metric', None),
            'metrics': checkpoint.get('metrics', {}),
            'model_version': self._compute_model_version(checkpoint),
        }

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"[INFERENCE_PATH] Predictor loaded: {self.model_path.name}, "
            f"params={total_params:,}, device={self.device}, "
            f"epoch={self.metadata['epoch']}"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _compute_model_version(checkpoint: dict) -> str:
        """Compute a short hash of model weights for version tracking (BP-3)."""
        state_dict = checkpoint.get('model_state_dict', {})
        h = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            h.update(key.encode())
            h.update(state_dict[key].cpu().numpy().tobytes())
        return h.hexdigest()[:12]

    def _load_checkpoint(self, path: Path) -> dict:
        """Load checkpoint with proper device mapping.

        Handles both formats:
            - Trainer checkpoint dict (has 'model_state_dict' key)
            - Raw state_dict (OrderedDict of weights)
        """
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle raw state_dict (no 'model_state_dict' key)
        if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
            # Check if it looks like a raw state_dict (keys are parameter names)
            if any(k.startswith(('mlp.', 'bigru.', 'attention.', 'output_head.')) for k in checkpoint.keys()):
                logger.info("[INFERENCE_PATH] Detected raw state_dict, wrapping")
                checkpoint = {'model_state_dict': checkpoint}

        logger.info(
            f"[CHECKPOINT] Model checkpoint loaded: {path} | "
            f"epoch={checkpoint.get('epoch', '?')} | "
            f"best_metric={checkpoint.get('best_metric', '?')}"
        )
        return checkpoint

    @staticmethod
    def _build_model(config: dict) -> DepressionModel:
        """Build DepressionModel from config dict."""
        return DepressionModel(
            input_dim=config.get('input_dim', 384),
            mlp_hidden=config.get('mlp', {}).get('hidden_dim', 15),
            mlp_dropout=config.get('mlp', {}).get('dropout', 0.5),
            mlp_bottleneck=config.get('mlp', {}).get('bottleneck', 20),
            gru_hidden=config.get('bigru', {}).get('hidden_size', 3),
            gru_layers=config.get('bigru', {}).get('num_layers', 1),
            head_dropout=config.get('head', {}).get('dropout', 0.5),
            gru_dropout=config.get('bigru', {}).get('dropout', 0.1),
            pooling=config.get('pooling', 'stats_direct'),
            stats_head_dim=config.get('stats_head_dim', 0),
            stats_mode=config.get('stats_mode', 'mean_std'),
        )

    @torch.no_grad()
    def predict(
        self,
        features: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray],
        lengths: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Run deterministic inference on PCA-reduced features.

        Accepts both numpy arrays and torch tensors.

        Args:
            features: (B, T, 64) — PCA-reduced, padded features
            mask: (B, T) — True for real chunks, False for padding
            lengths: (B,) — actual sequence lengths

        Returns:
            (B,) — predicted PHQ-8 scores (float tensor, on CPU)
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if isinstance(lengths, np.ndarray):
            lengths = torch.from_numpy(lengths)

        features = features.to(self.device)
        mask = mask.to(self.device)
        # lengths stay on CPU — required by pack_padded_sequence in BiGRU

        predictions = self.model(features, mask, lengths)
        # Clamp to valid PHQ-8 clinical range [0, 24]
        predictions = predictions.clamp(0.0, 24.0)
        return predictions.cpu()

    @torch.no_grad()
    def predict_single(self, features: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Predict PHQ-8 score for a single participant's features.

        Accepts both numpy arrays and torch tensors.

        Args:
            features: (T, 64) — PCA-reduced features for one participant

        Returns:
            float — predicted PHQ-8 score
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        T = features.shape[0]
        features = features.unsqueeze(0).to(self.device)  # (1, T, 64)
        mask = torch.ones(1, T, dtype=torch.bool, device=self.device)
        lengths = torch.tensor([T])  # CPU — required by pack_padded_sequence

        prediction = self.model(features, mask, lengths)
        # Clamp to valid PHQ-8 clinical range [0, 24]
        return float(prediction.clamp(0.0, 24.0).item())

    def predict_with_uncertainty(
        self,
        features: Union[torch.Tensor, np.ndarray],
        n_samples: int = 30,
    ) -> dict:
        """
        MC Dropout uncertainty estimation for a single participant (MED-1).

        Args:
            features: (T, 64) — PCA-reduced features
            n_samples: Number of MC forward passes

        Returns:
            Dict with 'mean', 'std', 'ci_lower', 'ci_upper' (all floats)
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        T = features.shape[0]
        features = features.unsqueeze(0).to(self.device)
        mask = torch.ones(1, T, dtype=torch.bool, device=self.device)
        lengths = torch.tensor([T])

        result = self.model.predict_with_uncertainty(features, mask, lengths, n_samples)
        return {
            'mean': float(result['mean'].item()),
            'std': float(result['std'].item()),
            'ci_lower': float(result['ci_lower'].item()),
            'ci_upper': float(result['ci_upper'].item()),
        }
