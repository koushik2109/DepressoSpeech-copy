"""
[LAYER_START] Session 6 (Revised): Attention Pooling
Mask-aware learnable attention over time steps → fixed-size representation.
Includes entropy tracking for regularization and dropout on attention weights.

[BOTH PATHS] Same architecture. Padding positions masked to -inf before softmax.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """
    Learnable additive attention for sequence pooling with entropy regularization.

    score_t = v^T · tanh(W · h_t)
    alpha = softmax(scores, masked)
    output = sum(alpha * h)

    Entropy regularization: H(alpha) = -sum(alpha * log(alpha))
    Higher entropy = attention spread across more chunks = better generalization.
    The attention_entropy attribute is accessible by the trainer for loss computation.

    Input:  (B, T, hidden_size)  e.g. (B, T, 8)
    Output: (B, hidden_size)     e.g. (B, 8) — single vector per subject
    """

    def __init__(self, hidden_size: int = 8, attn_dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # register_buffer ensures attention_entropy follows .to(device) automatically
        # Updated every forward pass; trainer reads it for entropy regularization
        self.register_buffer('attention_entropy', torch.tensor(0.0))

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[LAYER_START] AttentionPooling: dim={hidden_size}, "
            f"attn_dropout={attn_dropout}, params={num_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_size) — GRU output
            mask: (B, T) — True for real chunks, False for padding

        Returns:
            (B, hidden_size) — attention-weighted pooled representation
        """
        # Compute attention scores
        energy = torch.tanh(self.W(x))   # (B, T, hidden_size)
        scores = self.v(energy).squeeze(-1)  # (B, T)

        # Mask padding positions to -inf
        scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax over time dimension
        weights = torch.softmax(scores, dim=1)  # (B, T)

        # Handle edge case: all-padding rows
        weights = weights.masked_fill(weights.isnan(), 0.0)

        # Compute attention entropy: H = -sum(w * log(w)) for valid positions
        eps = 1e-8
        log_weights = torch.log(weights.clamp(min=eps))
        entropy_per_pos = -(weights * log_weights)  # (B, T)
        entropy_per_pos = entropy_per_pos.masked_fill(~mask, 0.0)
        valid_counts = mask.float().sum(dim=-1).clamp(min=1)  # (B,)
        # Normalize by log(T) to get entropy in [0, 1] range
        max_entropy = torch.log(valid_counts + eps)  # (B,)
        raw_entropy = entropy_per_pos.sum(dim=-1).clamp(min=0.0)  # (B,)
        self.attention_entropy = (raw_entropy / max_entropy.clamp(min=eps)).mean()

        # Dropout on attention weights (training only)
        weights = self.attn_dropout(weights)

        # Weighted sum: (B, 1, T) @ (B, T, hidden_size) → (B, hidden_size)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)

        return pooled
