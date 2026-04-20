"""
[LAYER_START] Session 6 (Revised): MLP Block
Two-layer bottleneck MLP for nonlinear feature transformation.

Single linear layer after PCA is mathematically redundant (two linear
transforms collapse into one). A bottleneck with nonlinearity allows
learning feature interactions that PCA cannot capture.

[BOTH PATHS] Same architecture. Dropout disabled in eval mode.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MLPBlock(nn.Module):
    """
    Two-layer bottleneck MLP: input → bottleneck → hidden_dim

    Architecture:
        Linear(input_dim → bottleneck) → GELU → Dropout →
        Linear(bottleneck → hidden_dim) → LayerNorm → Dropout → GELU

    Input:  (B, T, input_dim)   e.g. (B, T, 64)
    Output: (B, T, hidden_dim)  e.g. (B, T, 15)

    Why 2 layers: breaks the PCA→Linear redundancy.
    Why GELU: smoother gradients near zero, better for small models.
    Why bottleneck=20: keeps total model params under ~2,600.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 15,
        dropout: float = 0.5,
        bottleneck_dim: int = 20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weight norm on first layer constrains its effective capacity.
        # With 1,300 params (61% of model), this layer is the main memorization risk.
        # Weight norm decouples direction from magnitude, preventing any single
        # neuron from growing arbitrarily large to memorize specific participants.
        first_linear = nn.Linear(input_dim, bottleneck_dim)

        try:
            # PyTorch >= 2.0 preferred API
            normed_linear = nn.utils.parametrizations.weight_norm(first_linear)
        except AttributeError:
            # Fallback for PyTorch < 2.0
            normed_linear = nn.utils.weight_norm(first_linear)

        self.layers = nn.Sequential(
            normed_linear,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        self._init_weights()

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[LAYER_START] MLPBlock: {input_dim} → {bottleneck_dim} → {hidden_dim}, "
            f"dropout={dropout}, params={num_params:,}"
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            (B, T, hidden_dim)
        """
        return self.layers(x)
