"""
Gated Multimodal Fusion Model for Depression Severity Prediction.

Architecture:
  Text branch:   SBERT (384) → StatsPool(mean+std=768) → BN → Linear → 128
  Audio branch:  HuBERT (768) → StatsPool(mean+std=1536) → BN → Linear → 128
  Quality gate:  audio_quality scores → sigmoid gate per segment (before pooling)
  Fusion gate:   learned gate deciding text vs audio contribution
  Head:          fused(128) → Linear → 1

Key design principles:
  1. Text branch is pre-trained and frozen first (preserves CCC=0.54)
  2. Audio gate: quality scores suppress low-quality audio segments
  3. Fusion gate: learned scalar/vector gate prevents audio from hurting text
  4. Modality dropout: randomly zero one modality during training → robustness
  5. Fallback: if audio hurts → gate learns to shut it off → degrades to text-only
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional

from src.models.statistics_pooling import StatisticsPooling

logger = logging.getLogger(__name__)


class GatedMultimodalModel(nn.Module):
    """
    Gated multimodal fusion for depression severity prediction.

    Inputs:
        text_features:     (B, T, 384)  — SBERT embeddings per chunk
        audio_features:    (B, T, 768)  — HuBERT embeddings per chunk
        audio_quality:     (B, T)       — quality scores [0,1] per chunk
        mask:              (B, T)       — True for real chunks
        lengths:           (B,)         — actual sequence lengths

    Output:
        predictions:       (B,)         — PHQ-8 scores
    """

    def __init__(
        self,
        text_dim: int = 384,
        audio_dim: int = 768,
        proj_dim: int = 128,
        stats_mode: str = "mean_std",
        dropout: float = 0.1,
        modality_dropout: float = 0.15,
        quality_gate_bias: float = 0.0,
    ):
        """
        Args:
            text_dim: Input dimension of text features (SBERT)
            audio_dim: Input dimension of audio features (HuBERT)
            proj_dim: Projection dimension for each branch after pooling
            stats_mode: Statistics pooling mode ("mean_std" or "mean_std_min_max")
            dropout: Dropout rate for projection heads
            modality_dropout: Probability of dropping entire modality during training
            quality_gate_bias: Initial bias for quality gate (0=neutral, negative=skeptical)
        """
        super().__init__()

        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.proj_dim = proj_dim
        self.modality_dropout = modality_dropout

        # === Text Branch ===
        self.text_pool = StatisticsPooling(input_dim=text_dim, stats=stats_mode)
        text_pool_dim = self.text_pool.output_dim  # 384*2 = 768 for mean_std
        self.text_norm = nn.BatchNorm1d(text_pool_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_pool_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # === Audio Branch ===
        self.audio_pool = StatisticsPooling(input_dim=audio_dim, stats=stats_mode)
        audio_pool_dim = self.audio_pool.output_dim  # 768*2 = 1536 for mean_std
        self.audio_norm = nn.BatchNorm1d(audio_pool_dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_pool_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # === Quality Gate: suppress low-quality audio segments ===
        # Transforms per-segment quality score into a gate value
        # audio_features *= sigmoid(quality_gate(quality_score))
        self.quality_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )
        # Initialize quality gate to pass-through (output ~0.5 for quality=0.5)
        with torch.no_grad():
            self.quality_gate[0].weight.fill_(2.0)
            self.quality_gate[0].bias.fill_(quality_gate_bias)

        # === Fusion Gate: learned text vs audio weighting ===
        # Input: [text_proj, audio_proj] → scalar gate → fused = g*text + (1-g)*audio
        self.fusion_gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1),
            nn.Sigmoid(),
        )

        # === Prediction Head ===
        self.head = nn.Sequential(
            nn.Linear(proj_dim, 1),
        )

        # Training stage: controls forward path
        # "text_only" → bypass fusion gate, use text_proj directly
        # "audio_gate" → freeze text, train audio + gate
        # "joint" → full model, all unfrozen
        self._training_stage = "joint"

        # Initialize
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[LAYER_START] GatedMultimodalModel: "
            f"text({text_dim})→{text_pool_dim}→{proj_dim}, "
            f"audio({audio_dim})→{audio_pool_dim}→{proj_dim}, "
            f"fusion_gate→{proj_dim}→1, "
            f"total params={total_params:,}"
        )

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        audio_quality: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_features:  (B, T, 384)
            audio_features: (B, T, 768)
            audio_quality:  (B, T) quality scores [0, 1]
            mask:           (B, T) True for valid chunks
            lengths:        (B,) actual sequence lengths

        Returns:
            (B,) predicted PHQ-8 scores
        """
        B, T, _ = text_features.shape

        # === Stage 1 fast path: skip all audio computation ===
        if self._training_stage == "text_only":
            # Modality dropout on text (training only)
            if self.training and self.modality_dropout > 0:
                text_drop_rand = torch.rand(B, device=text_features.device)
                text_drop_mask = (text_drop_rand < self.modality_dropout * 0.5).unsqueeze(-1).unsqueeze(-1)
                text_input = text_features * (~text_drop_mask).float()
            else:
                text_input = text_features

            text_pooled = self.text_pool(text_input, mask)
            text_pooled = self.text_norm(text_pooled)
            text_proj = self.text_proj(text_pooled)
            self.last_gate_value = torch.tensor(1.0, device=text_features.device)
            return self.head(text_proj).squeeze(-1)

        # === Quality gate: modulate audio per-segment ===
        quality_expanded = audio_quality.unsqueeze(-1)  # (B, T, 1)
        quality_weights = self.quality_gate(quality_expanded)  # (B, T, 1)
        gated_audio = audio_features * quality_weights  # (B, T, 768)

        # === Modality dropout (training only) ===
        if self.training and self.modality_dropout > 0:
            drop_rand = torch.rand(B, device=text_features.device)
            # Drop audio for some samples (text-only fallback)
            audio_drop_mask = (drop_rand < self.modality_dropout).unsqueeze(-1).unsqueeze(-1)
            gated_audio = gated_audio * (~audio_drop_mask).float()

            # Drop text for some samples (less frequently, audio-only)
            text_drop_rand = torch.rand(B, device=text_features.device)
            text_drop_mask = (text_drop_rand < self.modality_dropout * 0.5).unsqueeze(-1).unsqueeze(-1)
            text_input = text_features * (~text_drop_mask).float()
        else:
            text_input = text_features

        # === Text branch: pool → project ===
        text_pooled = self.text_pool(text_input, mask)   # (B, 768)
        text_pooled = self.text_norm(text_pooled)
        text_proj = self.text_proj(text_pooled)           # (B, 128)

        # === Audio branch: pool → project ===
        audio_pooled = self.audio_pool(gated_audio, mask)  # (B, 1536)
        audio_pooled = self.audio_norm(audio_pooled)
        audio_proj = self.audio_proj(audio_pooled)          # (B, 128)

        # === Fusion gate ===
        gate_input = torch.cat([text_proj, audio_proj], dim=-1)  # (B, 256)
        gate = self.fusion_gate(gate_input)  # (B, 1) — how much text to use
        self.last_gate_value = gate.mean()

        fused = gate * text_proj + (1 - gate) * audio_proj  # (B, 128)

        # === Prediction ===
        predictions = self.head(fused).squeeze(-1)  # (B,)

        return predictions

    def set_training_stage(self, stage: str):
        """Set the training stage: 'text_only', 'audio_gate', or 'joint'."""
        assert stage in ("text_only", "audio_gate", "joint"), f"Invalid stage: {stage}"
        self._training_stage = stage
        logger.info(f"[TRAINING] Training stage set to: {stage}")

    def get_gate_stats(self) -> Dict[str, float]:
        """Return fusion gate statistics for monitoring."""
        if hasattr(self, 'last_gate_value'):
            return {'fusion_gate_mean': self.last_gate_value.item()}
        return {'fusion_gate_mean': 0.5}

    def freeze_text_branch(self):
        """Freeze text branch parameters (for staged training)."""
        for param in self.text_pool.parameters():
            param.requires_grad = False
        for param in self.text_norm.parameters():
            param.requires_grad = False
        for param in self.text_proj.parameters():
            param.requires_grad = False
        logger.info("[TRAINING] Text branch frozen")

    def unfreeze_text_branch(self):
        """Unfreeze text branch for joint fine-tuning."""
        for param in self.text_pool.parameters():
            param.requires_grad = True
        for param in self.text_norm.parameters():
            param.requires_grad = True
        for param in self.text_proj.parameters():
            param.requires_grad = True
        logger.info("[TRAINING] Text branch unfrozen")

    def freeze_audio_branch(self):
        """Freeze audio branch parameters."""
        for param in self.audio_pool.parameters():
            param.requires_grad = False
        for param in self.audio_norm.parameters():
            param.requires_grad = False
        for param in self.audio_proj.parameters():
            param.requires_grad = False
        for param in self.quality_gate.parameters():
            param.requires_grad = False
        logger.info("[TRAINING] Audio branch frozen")

    def unfreeze_audio_branch(self):
        """Unfreeze audio branch."""
        for param in self.audio_pool.parameters():
            param.requires_grad = True
        for param in self.audio_norm.parameters():
            param.requires_grad = True
        for param in self.audio_proj.parameters():
            param.requires_grad = True
        for param in self.quality_gate.parameters():
            param.requires_grad = True
        logger.info("[TRAINING] Audio branch unfrozen")
