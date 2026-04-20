"""
Statistics Pooling: aggregates variable-length sequences into fixed-size vectors
by computing per-feature statistics over valid timesteps.

Supports configurable stat types: mean, std, min, max.
Default "mean_std" captures the distributional info that best predicts depression.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class StatisticsPooling(nn.Module):
    """
    Computes statistics over the time dimension, respecting padding mask.

    Stats modes:
        "mean_std":          mean + std → D*2
        "mean_std_min_max":  mean + std + min + max → D*4

    Input:  (B, T, D)
    Output: (B, D * num_stats)

    No learnable parameters.
    """

    STAT_MULTIPLIERS = {"mean": 1, "mean_std": 2, "mean_std_min_max": 4}

    def __init__(self, input_dim: int, stats: str = "mean_std"):
        super().__init__()
        assert stats in self.STAT_MULTIPLIERS, f"Unknown stats mode: {stats}"
        self.input_dim = input_dim
        self.stats = stats
        self.output_dim = input_dim * self.STAT_MULTIPLIERS[stats]
        logger.info(
            f"[LAYER_START] StatisticsPooling: {input_dim} → {self.output_dim} "
            f"(mode={stats}), params=0"
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — sequence features
            mask: (B, T) — True for valid timesteps, False for padding

        Returns:
            (B, D * num_stats)
        """
        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
        lengths = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)

        # Mean
        x_mean = (x * mask_f).sum(dim=1) / lengths  # (B, D)

        if self.stats == "mean":
            return x_mean

        # Std (with Bessel correction, clamped for single-element seqs)
        diff_sq = ((x - x_mean.unsqueeze(1)) ** 2) * mask_f
        variance = diff_sq.sum(dim=1) / lengths.clamp(min=2)
        x_std = variance.sqrt()  # (B, D)

        parts = [x_mean, x_std]

        if self.stats == "mean_std_min_max":
            big_pos = torch.finfo(x.dtype).max
            big_neg = torch.finfo(x.dtype).min
            x_for_min = x.clone()
            x_for_min[~mask] = big_pos
            x_min = x_for_min.min(dim=1).values

            x_for_max = x.clone()
            x_for_max[~mask] = big_neg
            x_max = x_for_max.max(dim=1).values

            parts.extend([x_min, x_max])

        return torch.cat(parts, dim=-1)
