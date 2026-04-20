"""
Ensemble Predictor: Averages predictions across 5-fold CV models.

Provides better calibrated predictions and inter-model disagreement
as an additional uncertainty signal beyond MC Dropout.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.inference.predictor import Predictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Loads N fold models and averages their predictions.

    Uncertainty from two sources:
        1. MC Dropout within each model (intra-model)
        2. Inter-model disagreement across folds (inter-model)
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        n_folds: int = 5,
        device: str = "auto",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.n_folds = n_folds
        self.predictors: List[Predictor] = []

        for i in range(n_folds):
            ckpt_path = self.checkpoint_dir / f"best_model_fold{i}.pt"
            if ckpt_path.exists():
                self.predictors.append(Predictor(model_path=ckpt_path, device=device))
            else:
                logger.warning(f"Fold {i} checkpoint not found: {ckpt_path}")

        logger.info(
            f"EnsemblePredictor: loaded {len(self.predictors)}/{n_folds} fold models"
        )

    def predict_single(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Average prediction across all fold models.

        Returns:
            Dict with 'mean', 'std', 'predictions' (per-fold)
        """
        predictions = []
        for predictor in self.predictors:
            with torch.no_grad():
                pred = predictor.model(
                    features.to(predictor.device),
                    mask.to(predictor.device),
                    lengths,
                )
                if isinstance(pred, dict):
                    pred = pred['phq_total']
                predictions.append(pred.cpu())

        stacked = torch.stack(predictions, dim=0)  # (n_folds, B)
        return {
            'mean': stacked.mean(dim=0).clamp(0, 24),
            'std': stacked.std(dim=0),
            'predictions': stacked,
        }

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        mc_samples: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """
        Combined uncertainty: MC Dropout per model + inter-model disagreement.

        Returns:
            Dict with 'mean', 'mc_std' (avg intra-model), 'ensemble_std' (inter-model),
            'total_std', 'ci_lower', 'ci_upper'
        """
        fold_means = []
        fold_mc_stds = []

        for predictor in self.predictors:
            mc_result = predictor.model.predict_with_uncertainty(
                features.to(predictor.device),
                mask.to(predictor.device),
                lengths,
                n_samples=mc_samples,
            )
            fold_means.append(mc_result['mean'].cpu())
            fold_mc_stds.append(mc_result['std'].cpu())

        means = torch.stack(fold_means, dim=0)       # (n_folds, B)
        mc_stds = torch.stack(fold_mc_stds, dim=0)   # (n_folds, B)

        ensemble_mean = means.mean(dim=0)
        ensemble_std = means.std(dim=0)       # inter-model disagreement
        avg_mc_std = mc_stds.mean(dim=0)      # avg intra-model uncertainty

        # Total uncertainty: combine both sources
        total_std = torch.sqrt(ensemble_std ** 2 + avg_mc_std ** 2)

        ci_lower = (ensemble_mean - 1.96 * total_std).clamp(0, 24)
        ci_upper = (ensemble_mean + 1.96 * total_std).clamp(0, 24)

        return {
            'mean': ensemble_mean.clamp(0, 24),
            'mc_std': avg_mc_std,
            'ensemble_std': ensemble_std,
            'total_std': total_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }
