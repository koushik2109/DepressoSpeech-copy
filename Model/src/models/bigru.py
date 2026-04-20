"""
[LAYER_START] Session 6 (Revised): Bidirectional GRU Encoder
Captures temporal dependencies across chunks. GRU chosen over LSTM for
~25% fewer parameters (3 gates vs 4).

[BOTH PATHS] Same architecture. Handles variable-length sequences via pack_padded_sequence.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BiGRUEncoder(nn.Module):
    """
    Single-layer Bidirectional GRU with LayerNorm and Dropout.

    Input:  (B, T, input_dim)     e.g. (B, T, 16) from MLP
    Output: (B, T, output_dim)    e.g. (B, T, 8)  — fwd+bwd concatenated

    Why GRU over LSTM:
        - 3 gates vs 4 → 25% fewer params per hidden unit
        - Comparable performance on small datasets
        - Simpler gradient flow

    No separate projection needed — output_dim = hidden_size * 2 directly.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_size: int = 4,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = hidden_size * 2  # bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # No effect with 1 layer
        )

        # Normalize + regularize output
        self.norm_dropout = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Dropout(dropout),
        )

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[LAYER_START] BiGRUEncoder: {input_dim} → BiGRU({hidden_size}×2) "
            f"→ {self.output_dim}, layers={num_layers}, dropout={dropout}, "
            f"params={num_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) — MLP output
            lengths: (B,) — original sequence lengths for packing
            mask: (B, T) — not used directly, lengths used for packing

        Returns:
            (B, T, output_dim) where output_dim = hidden_size * 2
        """
        B, T, _ = x.shape

        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=True
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            gru_out, _ = self.gru(x)

        # LayerNorm + Dropout: (B, T, output_dim)
        output = self.norm_dropout(gru_out)
        return output
