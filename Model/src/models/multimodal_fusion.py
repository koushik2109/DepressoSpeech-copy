"""
Multimodal Feature Fusion for Depression Severity Prediction.

Architecture (Feature-Concat + Additive Residual):
    Text:       (B,N,384) → StatsPool → (B,768) → BN → Linear → text_pred   [FROZEN]
    MFCC:       (B,N,120) ─┐
    eGeMAPS:    (B,N,88)  ─┤→ cat → (B,N,208) → AudioEncoder → (B,64) ─┐
    Behavioral: (B,16) ─────────────────────────────────────────────────┤→ cat → (B,80)
                                                    → BN → Linear(80,1) → residual
    Fusion:     output = text_pred + residual       [residual starts ≈ 0]

AudioEncoder uses Conv1d temporal modeling instead of raw stats pooling,
capturing local patterns in audio features before summarizing.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path

from src.models.statistics_pooling import StatisticsPooling

logger = logging.getLogger(__name__)

BEHAVIORAL_DIM = 16  # interview-level features from transcripts


class AudioEncoder(nn.Module):
    """
    Lightweight temporal encoder for audio features (MFCC + eGeMAPS).

    Conv1d(208, 64, k=3) → ReLU → Conv1d(64, 32, k=3) → ReLU
    → StatsPool(mean+std → 64) → LayerNorm(64)

    ~20K params. Captures local temporal patterns before pooling.
    """

    def __init__(self, input_dim: int = 208, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = StatisticsPooling(input_dim=out_dim, stats="mean_std")
        self.norm = nn.LayerNorm(self.pool.output_dim)  # 64

    @property
    def output_dim(self):
        return self.pool.output_dim  # 64

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 208) audio features
            mask: (B, N) bool
        Returns:
            (B, 64) pooled audio representation
        """
        # Conv1d expects (B, C, T)
        h = x.transpose(1, 2)  # (B, 208, N)
        h = self.relu(self.conv1(h))  # (B, 64, N)
        h = self.relu(self.conv2(h))  # (B, 32, N)
        h = h.transpose(1, 2)  # (B, N, 32)
        h = self.pool(h, mask)  # (B, 64)
        return self.norm(h)


class MultimodalFusion(nn.Module):

    TEXT_DIM = 384
    MFCC_DIM = 120
    EGEMAPS_DIM = 88
    AUDIO_DIM = MFCC_DIM + EGEMAPS_DIM  # 208

    def __init__(self, stats_mode: str = "mean_std", use_behavioral: bool = True,
                 residual_dropout: float = 0.0, use_audio_encoder: bool = True):
        super().__init__()
        self.use_behavioral = use_behavioral
        self.use_audio_encoder = use_audio_encoder

        # --- Text branch (frozen after loading pretrained weights) ---
        self.text_pool = StatisticsPooling(self.TEXT_DIM, stats=stats_mode)
        text_pooled_dim = self.text_pool.output_dim  # 768
        self.text_bn = nn.BatchNorm1d(text_pooled_dim)
        self.text_head = nn.Linear(text_pooled_dim, 1)

        # --- Audio branch ---
        if use_audio_encoder:
            self.audio_encoder = AudioEncoder(input_dim=self.AUDIO_DIM)
            audio_out_dim = self.audio_encoder.output_dim  # 64
        else:
            self.audio_pool = StatisticsPooling(self.AUDIO_DIM, stats=stats_mode)
            audio_out_dim = self.audio_pool.output_dim  # 416

        residual_dim = audio_out_dim + (BEHAVIORAL_DIM if use_behavioral else 0)

        self.residual_bn = nn.BatchNorm1d(residual_dim)
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0 else nn.Identity()
        self.residual_head = nn.Linear(residual_dim, 1)

        # Initialize residual head near zero so model starts at text_pred
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self._text_frozen = False

    def load_pretrained_text(self, checkpoint_path: str):
        """Load pretrained text-only model weights and freeze the text branch."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        loaded_any = False

        def _copy_param(target, key: str) -> bool:
            src = state.get(key)
            if src is None:
                logger.warning(f"Text preload skipped: missing key '{key}'")
                return False
            if tuple(src.shape) != tuple(target.shape):
                logger.warning(
                    "Text preload skipped: shape mismatch for '%s' (checkpoint=%s, model=%s)",
                    key,
                    tuple(src.shape),
                    tuple(target.shape),
                )
                return False
            target.copy_(src)
            return True

        loaded_any |= _copy_param(self.text_bn.weight.data, "pool_norm.weight")
        loaded_any |= _copy_param(self.text_bn.bias.data, "pool_norm.bias")
        loaded_any |= _copy_param(self.text_bn.running_mean, "pool_norm.running_mean")
        loaded_any |= _copy_param(self.text_bn.running_var, "pool_norm.running_var")
        loaded_any |= _copy_param(self.text_bn.num_batches_tracked, "pool_norm.num_batches_tracked")
        loaded_any |= _copy_param(self.text_head.weight.data, "output_head.weight")
        loaded_any |= _copy_param(self.text_head.bias.data, "output_head.bias")

        if loaded_any:
            self.freeze_text()
            ccc = ckpt.get("metrics", {}).get("ccc", "?")
            logger.info(f"Loaded pretrained text model (CCC={ccc}), text branch frozen")
        else:
            logger.warning("Skipped text checkpoint preload due to incompatible checkpoint shapes/keys")

    def freeze_text(self):
        self._text_frozen = True
        self.text_bn.eval()
        for p in self.text_bn.parameters():
            p.requires_grad = False
        for p in self.text_head.parameters():
            p.requires_grad = False

    def unfreeze_text(self):
        self._text_frozen = False
        for p in self.text_bn.parameters():
            p.requires_grad = True
        for p in self.text_head.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self._text_frozen:
            self.text_bn.eval()
        return self

    def forward(self, text, mfcc, egemaps, mask, behavioral=None, text_scale: float = 0.6):
        """
        Args:
            text:       (B, N, 384)
            mfcc:       (B, N, 120)
            egemaps:    (B, N, 88)
            mask:       (B, N) bool
            behavioral: (B, 16) or None — interview-level features
            text_scale: Weight applied to text_pred before adding residual.
                        Default 0.6 to reduce text dominance and give audio
                        branch more relative influence. Range [0.0, 1.0].
        Returns:
            (B, 1) predicted PHQ-8 scores
        """
        # Text pathway (frozen)
        t = self.text_pool(text, mask)
        t = self.text_bn(t)
        text_pred = self.text_head(t)               # (B, 1)

        # Audio pathway
        audio = torch.cat([mfcc, egemaps], dim=-1)  # (B, N, 208)
        if self.use_audio_encoder:
            a = self.audio_encoder(audio, mask)      # (B, 64)
        else:
            a = self.audio_pool(audio, mask)          # (B, 416)

        # Concat behavioral if available
        if self.use_behavioral and behavioral is not None:
            a = torch.cat([a, behavioral], dim=-1)   # (B, 432)

        # Residual prediction
        a = self.residual_bn(a)
        a = self.residual_dropout(a)
        residual = self.residual_head(a)             # (B, 1)

        # Scale text contribution so audio/behavioral residual has more influence
        return text_scale * text_pred + residual

    def predict_text_only(self, text, mask):
        t = self.text_pool(text, mask)
        t = self.text_bn(t)
        return self.text_head(t)

    def param_summary(self):
        def _count(module, trainable_only=False):
            return sum(
                p.numel() for p in module.parameters()
                if not trainable_only or p.requires_grad
            )

        text_total = _count(self.text_bn) + _count(self.text_head)
        text_train = _count(self.text_bn, True) + _count(self.text_head, True)

        if self.use_audio_encoder:
            audio_mod = self.audio_encoder
        else:
            audio_mod = self.audio_pool
        res_total = _count(audio_mod) + _count(self.residual_bn) + _count(self.residual_head)
        res_train = _count(audio_mod, True) + _count(self.residual_bn, True) + _count(self.residual_head, True)

        return {
            "text_total": text_total,
            "text_trainable": text_train,
            "audio_total": res_total,
            "audio_trainable": res_train,
            "total": text_total + res_total,
            "trainable": text_train + res_train,
            "audio_text_ratio": f"{res_train / max(text_total, 1):.2f}:1",
        }
