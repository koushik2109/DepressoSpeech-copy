"""
[LAYER_START] Session 8: Depression Severity Model
Supports two pooling modes:
  - "attention": MLP → BiGRU → Attention → GatedSkip → Linear (original)
  - "stats": MLP → BiGRU → StatisticsPooling(mean+std+min+max) → Linear

Statistics pooling captures distributional features (variability, extremes)
that are far more predictive of depression than learned attention weights
on this small dataset (163 subjects).

Expects PCA-reduced input (default 24-dim).
"""

import torch
import torch.nn as nn
import logging
from typing import Dict

from src.models.mlp_block import MLPBlock
from src.models.bigru import BiGRUEncoder
from src.models.attention import AttentionPooling
from src.models.statistics_pooling import StatisticsPooling

logger = logging.getLogger(__name__)


class DepressionModel(nn.Module):
    """
    Depression severity prediction model.

    Architecture (pooling="stats_direct", recommended for small datasets):
        Input (B, T, D) → StatsPool(mean+std+min+max) → MLP → 1
        No temporal layers — raw feature distributions predict depression best.

    Architecture (pooling="stats"):
        Input (B, T, D) → MLP → BiGRU → StatsPool → MLP → 1

    Architecture (pooling="attention"):
        Input (B, T, D) → MLP → BiGRU → Attention → GatedSkip → Linear → 1
    """

    def __init__(
        self,
        input_dim: int = 24,
        mlp_hidden: int = 12,
        mlp_dropout: float = 0.1,
        mlp_bottleneck: int = 16,
        gru_hidden: int = 6,
        gru_layers: int = 1,
        head_dropout: float = 0.1,
        gru_dropout: float = 0.1,
        pooling: str = "stats_direct",
        stats_head_dim: int = 0,
        stats_mode: str = "mean_std",
        multitask: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.pooling_type = pooling
        self.multitask = multitask
        self.gru_output_dim = gru_hidden * 2  # bidirectional

        if pooling == "stats_direct":
            # Direct statistics on raw input — no temporal layers
            self.stats_pool = StatisticsPooling(input_dim=input_dim, stats=stats_mode)
            pool_out_dim = self.stats_pool.output_dim

            # BatchNorm standardizes pooled features — critical for linear head
            # With full-batch training (batch_size=163), acts like StandardScaler
            self.pool_norm = nn.BatchNorm1d(pool_out_dim)

            if stats_head_dim > 0:
                # MLP head: pool → norm → hidden → 1
                self.output_head = nn.Sequential(
                    nn.Linear(pool_out_dim, stats_head_dim),
                    nn.ReLU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(stats_head_dim, 1),
                )
            else:
                # Linear head: pool → norm → 1 (best for high-dim text features)
                self.output_head = nn.Linear(pool_out_dim, 1)
        else:
            # Shared: MLP + BiGRU for temporal modes
            self.mlp = MLPBlock(
                input_dim=input_dim,
                hidden_dim=mlp_hidden,
                dropout=mlp_dropout,
                bottleneck_dim=mlp_bottleneck,
            )
            self.bigru = BiGRUEncoder(
                input_dim=mlp_hidden,
                hidden_size=gru_hidden,
                num_layers=gru_layers,
                dropout=gru_dropout,
            )

            if pooling == "stats":
                self.stats_pool = StatisticsPooling(
                    input_dim=self.gru_output_dim, stats=stats_mode
                )
                pool_out_dim = self.stats_pool.output_dim
                self.output_head = nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(pool_out_dim, stats_head_dim if stats_head_dim > 0 else pool_out_dim),
                    nn.ReLU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(stats_head_dim if stats_head_dim > 0 else pool_out_dim, 1),
                )
            else:
                # Attention pooling path (original architecture)
                self.attention = AttentionPooling(hidden_size=self.gru_output_dim)
                self.skip_proj = nn.Linear(mlp_hidden, self.gru_output_dim)
                self.skip_gate = nn.Sequential(
                    nn.Linear(self.gru_output_dim * 2, 1),
                    nn.Sigmoid(),
                )
                self.output_head = nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(self.gru_output_dim, 1),
                )
                pool_out_dim = self.gru_output_dim

        # Multi-task heads (optional, for richer supervision)
        if multitask:
            # Determine pooled dim for multi-task heads
            if pooling == "stats_direct":
                mt_dim = self.stats_pool.output_dim
            elif pooling == "stats":
                mt_dim = self.stats_pool.output_dim
            else:
                mt_dim = self.gru_output_dim
            self.item_head = nn.Linear(mt_dim, 8)    # per-item PHQ-8 scores (0-3 each)
            self.binary_head = nn.Linear(mt_dim, 1)  # depression yes/no

        # MC Dropout — applied only during predict_with_uncertainty()
        # No learnable params, so existing checkpoints load fine
        self.mc_dropout = nn.Dropout(p=0.15)

        # Initialize weights
        self._init_weights()

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[LAYER_START] DepressionModel(pooling={pooling}, stats={stats_mode}): "
            f"{input_dim}→{pooling}→{pool_out_dim if pooling.startswith('stats') else 'attn'}→1, "
            f"total params={total_params:,}"
        )

    def _init_weights(self):
        """Xavier for linear, orthogonal for GRU."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name and param.dim() >= 2:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _get_pooled(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Extract pooled representation before the output head.

        Returns:
            (B, pool_dim) — pooled features ready for the output head
        """
        if self.pooling_type == "stats_direct":
            pooled = self.stats_pool(features, mask)
            pooled = self.pool_norm(pooled)
        elif self.pooling_type == "stats":
            mlp_out = self.mlp(features)
            gru_out = self.bigru(mlp_out, lengths)
            pooled = self.stats_pool(gru_out, mask)
        else:
            mlp_out = self.mlp(features)
            gru_out = self.bigru(mlp_out, lengths)

            mask_expanded = mask.unsqueeze(-1).float()
            mlp_sum = (mlp_out * mask_expanded).sum(dim=1)
            mlp_mean = mlp_sum / mask_expanded.sum(dim=1).clamp(min=1)
            skip = self.skip_proj(mlp_mean)

            attended = self.attention(gru_out, mask)

            gate = self.skip_gate(torch.cat([attended, skip], dim=-1))
            self.last_gate_value = gate.mean()
            pooled = gate * attended + (1 - gate) * skip

        return pooled

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, input_dim) — PCA-reduced, padded features
            mask: (B, T) — True for real chunks, False for padding
            lengths: (B,) — actual sequence lengths (sorted descending)

        Returns:
            (B,) — predicted PHQ-8 scores
        """
        pooled = self._get_pooled(features, mask, lengths)
        phq_total = self.output_head(pooled).squeeze(-1)

        if self.multitask:
            return {
                'phq_total': phq_total,
                'phq_items': self.item_head(pooled),      # (B, 8)
                'binary': self.binary_head(pooled).squeeze(-1),  # (B,)
            }
        return phq_total

    def get_attention_entropy(self) -> torch.Tensor:
        """Return current attention entropy for loss regularization.
        Returns 0 if using statistics pooling (no attention weights).
        """
        if self.pooling_type == "stats":
            return torch.tensor(0.0)
        return self.attention.attention_entropy

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        n_samples: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """
        MC Dropout uncertainty estimation (MED-1).

        Runs forward pass n_samples times with dropout enabled to estimate
        prediction uncertainty. Returns mean, std, and confidence interval.

        Args:
            features: (B, T, input_dim)
            mask: (B, T)
            lengths: (B,)
            n_samples: Number of MC forward passes (default: 30)

        Returns:
            Dict with keys: 'mean', 'std', 'ci_lower', 'ci_upper', 'samples'
        """
        self.eval()
        with torch.no_grad():
            pooled = self._get_pooled(features, mask, lengths)

        # MC Dropout: apply dropout to pooled features for each sample
        self.mc_dropout.train()  # Force dropout active
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                dropped = self.mc_dropout(pooled)
                pred = self.output_head(dropped).squeeze(-1)
                samples.append(pred)

        self.eval()  # Restore eval mode
        samples_t = torch.stack(samples, dim=0)  # (n_samples, B)
        mean = samples_t.mean(dim=0)
        std = samples_t.std(dim=0)

        # 95% confidence interval (approx 1.96 * std)
        ci_lower = (mean - 1.96 * std).clamp(0.0, 24.0)
        ci_upper = (mean + 1.96 * std).clamp(0.0, 24.0)

        return {
            'mean': mean.clamp(0.0, 24.0),
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'samples': samples_t,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter counts per sub-module."""
        counts = {}
        if self.pooling_type == "stats_direct":
            counts['stats_pool'] = 0
            counts['output_head'] = sum(p.numel() for p in self.output_head.parameters() if p.requires_grad)
        elif self.pooling_type == "stats":
            counts['mlp'] = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
            counts['bigru'] = sum(p.numel() for p in self.bigru.parameters() if p.requires_grad)
            counts['stats_pool'] = 0
            counts['output_head'] = sum(p.numel() for p in self.output_head.parameters() if p.requires_grad)
        else:
            counts['mlp'] = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
            counts['bigru'] = sum(p.numel() for p in self.bigru.parameters() if p.requires_grad)
            counts['attention'] = sum(p.numel() for p in self.attention.parameters() if p.requires_grad)
            counts['skip_proj'] = sum(p.numel() for p in self.skip_proj.parameters() if p.requires_grad)
            counts['skip_gate'] = sum(p.numel() for p in self.skip_gate.parameters() if p.requires_grad)
            counts['output_head'] = sum(p.numel() for p in self.output_head.parameters() if p.requires_grad)
        counts['total'] = sum(counts.values())
        return counts
