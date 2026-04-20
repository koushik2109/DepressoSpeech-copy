"""
Model Architecture (Session 6 — Revised)
MLP + BiGRU + Attention Pooling + Output Head for depression severity prediction.
~1,614 params designed for 163-sample training set.
"""
from .mlp_block import MLPBlock
from .bigru import BiGRUEncoder
from .attention import AttentionPooling
from .depression_model import DepressionModel

__all__ = [
    "MLPBlock",
    "BiGRUEncoder",
    "AttentionPooling",
    "DepressionModel",
]
