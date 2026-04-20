"""
[LAYER_START] Session 7: Training Metrics
CCC (primary), RMSE, MAE for depression severity evaluation.

[TRAINING PATH] Computed per epoch on train and validation sets.
"""

import torch
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def concordance_correlation_coefficient(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Concordance Correlation Coefficient (CCC).

    Measures agreement between predicted and true PHQ-8 scores.
    Range: [-1, 1]. 1 = perfect agreement.

    CCC = (2 * rho * sigma_p * sigma_t) /
          (sigma_p^2 + sigma_t^2 + (mu_p - mu_t)^2)

    Args:
        predictions: (N,) predicted PHQ-8 scores
        targets: (N,) ground truth PHQ-8 scores

    Returns:
        CCC value (float)
    """
    if len(predictions) < 2:
        return 0.0

    mu_p = predictions.mean()
    mu_t = targets.mean()
    sigma_p = predictions.std(ddof=0)
    sigma_t = targets.std(ddof=0)

    if sigma_p < 1e-8 and sigma_t < 1e-8:
        return 1.0 if np.abs(mu_p - mu_t) < 1e-8 else 0.0

    covariance = np.mean((predictions - mu_p) * (targets - mu_t))

    denominator = sigma_p**2 + sigma_t**2 + (mu_p - mu_t)**2
    if denominator < 1e-8:
        return 0.0

    ccc = (2 * covariance) / denominator
    return float(ccc)


def root_mean_squared_error(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """RMSE between predicted and true PHQ-8 scores."""
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def mean_absolute_error(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """MAE between predicted and true PHQ-8 scores."""
    return float(np.mean(np.abs(predictions - targets)))


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predictions: (N,) predicted PHQ-8 scores
        targets: (N,) ground truth PHQ-8 scores

    Returns:
        Dict with keys: 'ccc', 'rmse', 'mae'
    """
    metrics = {
        'ccc': concordance_correlation_coefficient(predictions, targets),
        'rmse': root_mean_squared_error(predictions, targets),
        'mae': mean_absolute_error(predictions, targets),
    }
    return metrics
