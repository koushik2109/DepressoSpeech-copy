"""
[LAYER_START] Session 7: Training Losses
Addresses PHQ-8 label imbalance and metric-loss mismatch.

WeightedMSELoss: Up-weights moderate/severe depression samples (binary).
ContinuousWeightedLoss: Inverse-frequency weighting by PHQ-8 bin (smooth).
CCCLoss: Directly optimizes Concordance Correlation Coefficient.
CombinedLoss: Weighted sum of ContinuousWeightedMSE + CCC for balanced optimization.

[TRAINING PATH] Used during training only. Inference uses no loss.
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class WeightedMSELoss(nn.Module):
    """
    MSE loss with per-sample weights based on PHQ-8 severity.

    Samples with PHQ-8 >= threshold get higher weight to address
    class imbalance (most subjects have low scores in E-DAIC-WOZ).

    L = (1/N) * sum(w_i * (pred_i - target_i)^2)

    where w_i = high_weight if target_i >= threshold else low_weight
    """

    def __init__(
        self,
        phq_threshold: float = 10.0,
        high_weight: float = 2.0,
        low_weight: float = 1.0,
    ):
        super().__init__()
        self.phq_threshold = phq_threshold
        self.high_weight = high_weight
        self.low_weight = low_weight

        logger.info(
            f"[LAYER_START] WeightedMSELoss: threshold={phq_threshold}, "
            f"high_weight={high_weight}, low_weight={low_weight}"
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B,) predicted PHQ-8 scores
            targets: (B,) ground truth PHQ-8 scores

        Returns:
            Scalar weighted MSE loss
        """
        # Per-sample weights based on severity
        weights = torch.where(
            targets >= self.phq_threshold,
            torch.tensor(self.high_weight, device=targets.device, dtype=targets.dtype),
            torch.tensor(self.low_weight, device=targets.device, dtype=targets.dtype),
        )

        # Weighted squared error
        squared_error = (predictions - targets) ** 2
        weighted_loss = (weights * squared_error).mean()

        return weighted_loss


class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient loss.

    Directly optimizes the evaluation metric (CCC) to eliminate
    the train-on-MSE / evaluate-on-CCC mismatch.

    CCC = (2 * cov(pred, target)) / (var_pred + var_target + (mean_pred - mean_target)^2)
    Loss = 1 - CCC  (minimize this)

    Range: [0, 2] where 0 = perfect agreement, 2 = perfect anti-correlation.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        logger.info("[LAYER_START] CCCLoss initialized")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B,) predicted PHQ-8 scores
            targets: (B,) ground truth PHQ-8 scores

        Returns:
            Scalar 1-CCC loss
        """
        predictions = predictions.squeeze()

        # BP-6: CCC is unreliable with fewer than 4 samples; fall back to MSE
        if predictions.numel() < 4:
            return torch.nn.functional.mse_loss(predictions, targets) / (24.0 ** 2)

        pred_mean = predictions.mean()
        target_mean = targets.mean()
        pred_var = predictions.var(unbiased=False)
        target_var = targets.var(unbiased=False)
        covariance = ((predictions - pred_mean) * (targets - target_mean)).mean()

        ccc = (2.0 * covariance) / (
            pred_var + target_var + (pred_mean - target_mean) ** 2 + self.eps
        )

        return 1.0 - ccc


class ContinuousWeightedLoss(nn.Module):
    """
    MSE loss with continuous inverse-frequency weighting by PHQ-8 bin.

    Instead of binary threshold (depressed/not), assigns weight
    proportional to how rare each score range is in the training set.

    Weight formula per bin:
        w(bin_i) = 1 / (count(bin_i) * n_bins), normalized to mean=1

    This ensures every PHQ-8 severity range contributes equally to the loss,
    regardless of how many participants fall in that range.

    Why this beats binary weighting:
    - PHQ=0 (very common) gets low weight
    - PHQ=15 (very rare) gets high weight
    - PHQ=9 (borderline, also rare) gets appropriate weight
    - Smooth weighting prevents learning a step function at threshold=10

    Call .fit(train_labels) before training. If not called, uniform weights.
    """

    def __init__(
        self,
        n_bins: int = 5,
        floor_weight: float = 0.5,
        ceil_weight: float = 5.0,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.floor_weight = floor_weight
        self.ceil_weight = ceil_weight

        # register_buffer: these tensors follow .to(device) automatically
        self.register_buffer('bin_weights_t', torch.ones(n_bins))
        self.register_buffer('bin_edges_t', torch.linspace(0, 24, n_bins + 1))
        self._fitted = False

        logger.info(
            f"[LAYER_START] ContinuousWeightedLoss: n_bins={n_bins}, "
            f"floor={floor_weight}, ceil={ceil_weight}"
        )

    def fit(self, train_labels):
        """Compute bin weights from training label distribution.

        Call once before training starts.

        Args:
            train_labels: list or array of PHQ-8 scores (0-24)
        """
        labels = np.array(train_labels, dtype=np.float32)
        bin_edges = np.linspace(0, 24, self.n_bins + 1)
        bin_counts = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == self.n_bins - 1:
                mask = (labels >= low) & (labels <= high)
            else:
                mask = (labels >= low) & (labels < high)
            bin_counts[i] = max(mask.sum(), 1)  # avoid division by zero

        # Inverse frequency weighting, normalized to mean=1
        weights = 1.0 / bin_counts
        weights = weights / weights.sum() * self.n_bins

        # Clamp to prevent extreme weights
        weights = np.clip(weights, self.floor_weight, self.ceil_weight)

        # Store as registered buffers (follow .to(device) automatically)
        self.bin_edges_t.copy_(torch.from_numpy(bin_edges.astype(np.float32)))
        self.bin_weights_t.copy_(torch.from_numpy(weights.astype(np.float32)))
        self._fitted = True

        logger.info(f"[LOSS] Continuous bin edges: {bin_edges.tolist()}")
        logger.info(f"[LOSS] Continuous bin counts: {bin_counts.tolist()}")
        logger.info(f"[LOSS] Continuous bin weights: {weights.tolist()}")

    def _get_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Map each target value to its bin weight (vectorized, no Python loops)."""
        if not self._fitted:
            return torch.ones_like(targets)

        # torch.bucketize: maps each target to its bin index in a single GPU kernel
        # bin_edges_t[1:-1] gives the interior edges for bucket boundaries
        bin_indices = torch.bucketize(targets, self.bin_edges_t[1:-1])
        bin_indices = bin_indices.clamp(0, self.n_bins - 1)

        # Fancy indexing: one GPU kernel for all samples
        return self.bin_weights_t[bin_indices]

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B,) predicted PHQ-8 scores
            targets: (B,) ground truth PHQ-8 scores

        Returns:
            Scalar weighted MSE loss
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        weights = self._get_weights(targets)
        squared_error = (predictions - targets) ** 2
        return (weights * squared_error).mean()


class CombinedLoss(nn.Module):
    """
        Combined loss: normalized ContinuousWeightedMSE plus weighted CCC loss.

        L_total = (L_cwmse / 24^2) + ccc_weight * L_ccc

    - ContinuousWeightedMSE: inverse-frequency bin weighting ensures all PHQ-8
      severity ranges contribute equally, regardless of class imbalance.
    - CCCLoss: aligns optimization directly with the evaluation metric.
        - ccc_weight controls the contribution of CCC relative to normalized MSE.

    IMPORTANT: Call .fit(train_labels) before training to compute bin weights.
    If not called, falls back to uniform weights (equivalent to plain MSE).
    """

    def __init__(
        self,
        phq_threshold: float = 10.0,
        high_weight: float = 2.0,
        low_weight: float = 1.0,
        ccc_weight: float = 0.5,
        n_bins: int = 5,
        floor_weight: float = 0.5,
        ceil_weight: float = 5.0,
    ):
        super().__init__()
        self.cwmse = ContinuousWeightedLoss(
            n_bins=n_bins,
            floor_weight=floor_weight,
            ceil_weight=ceil_weight,
        )
        self.ccc_loss = CCCLoss()
        self.ccc_weight = ccc_weight
        # Keep legacy wmse for backward compat but don't use it in forward
        self._legacy_wmse = WeightedMSELoss(phq_threshold, high_weight, low_weight)

        self._fit_warned = False  # Track if we've warned about missing fit()

        logger.info(
            f"[LAYER_START] CombinedLoss: ContinuousWeightedMSE + {ccc_weight}*CCC, "
            f"n_bins={n_bins}, floor={floor_weight}, ceil={ceil_weight}"
        )

    def fit(self, train_labels):
        """Compute continuous bin weights from training label distribution.

        Call once before training starts. If not called, uniform weights are used.

        Args:
            train_labels: list/array of PHQ-8 scores from training set
        """
        self.cwmse.fit(train_labels)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B,) predicted PHQ-8 scores
            targets: (B,) ground truth PHQ-8 scores

        Returns:
            Scalar combined loss
        """
        # Warn once if fit() was never called (silent failure protection)
        if not self._fit_warned and not self.cwmse._fitted:
            logger.warning(
                "[LOSS] CombinedLoss.fit() was never called — using uniform "
                "weights. Call criterion.fit(train_labels) before training "
                "for proper imbalance handling."
            )
            self._fit_warned = True

        mse_loss = self.cwmse(predictions, targets)
        ccc_loss = self.ccc_loss(predictions, targets)

        # Normalize MSE to ~1.0 scale so CCC and entropy terms aren't dwarfed.
        # PHQ-8 range is 0-24, so max squared error is 576. Typical MSE ≈ 20-50.
        # Dividing by 24² normalizes to [0, 1] range, matching CCC loss scale.
        mse_normalized = mse_loss / (24.0 ** 2)

        return mse_normalized + self.ccc_weight * ccc_loss


class MultiTaskLoss(nn.Module):
    """
    Wraps a primary loss (on total PHQ-8) with auxiliary per-item and binary losses.

    L = L_primary + item_weight * MSE(item_preds, item_targets) + binary_weight * BCE(binary_pred, binary_target)
    """

    def __init__(self, primary_loss: nn.Module, item_weight: float = 0.1, binary_weight: float = 0.05):
        super().__init__()
        self.primary_loss = primary_loss
        self.item_weight = item_weight
        self.binary_weight = binary_weight
        self.item_criterion = nn.MSELoss()
        self.binary_criterion = nn.BCEWithLogitsLoss()

        logger.info(
            f"[LAYER_START] MultiTaskLoss: item_weight={item_weight}, binary_weight={binary_weight}"
        )

    def fit(self, train_labels):
        """Delegate fit to primary loss if it supports it."""
        if hasattr(self.primary_loss, 'fit'):
            self.primary_loss.fit(train_labels)

    def forward(self, model_output, targets, item_labels=None, binary_labels=None):
        """
        Args:
            model_output: dict with 'phq_total', 'phq_items', 'binary' OR tensor (B,)
            targets: (B,) total PHQ-8 scores
            item_labels: (B, 8) per-item scores (0-3)
            binary_labels: (B,) binary depression labels
        """
        if isinstance(model_output, dict):
            predictions = model_output['phq_total']
        else:
            predictions = model_output

        loss = self.primary_loss(predictions, targets)

        if isinstance(model_output, dict) and item_labels is not None:
            loss = loss + self.item_weight * self.item_criterion(model_output['phq_items'], item_labels)

        if isinstance(model_output, dict) and binary_labels is not None:
            loss = loss + self.binary_weight * self.binary_criterion(model_output['binary'], binary_labels)

        return loss


class AdaptiveCombinedLoss(nn.Module):
    """
    MSE + CCC loss with CCC warmup schedule.

    Starts with pure MSE and gradually introduces CCC loss over warmup_epochs.
    This stabilizes early training (when predictions are far off) while
    eventually optimizing directly for the evaluation metric.
    """

    def __init__(self, mse_loss: nn.Module, ccc_max_weight: float = 1.0, warmup_epochs: int = 50):
        super().__init__()
        self.mse_loss = mse_loss
        self.ccc_loss = CCCLoss()
        self.ccc_max_weight = ccc_max_weight
        self.warmup_epochs = warmup_epochs
        self.ccc_weight = 0.0

        logger.info(
            f"[LAYER_START] AdaptiveCombinedLoss: ccc_max_weight={ccc_max_weight}, warmup={warmup_epochs}"
        )

    def fit(self, train_labels):
        """Delegate fit to MSE loss if it supports it."""
        if hasattr(self.mse_loss, 'fit'):
            self.mse_loss.fit(train_labels)

    def set_epoch(self, epoch: int):
        """Update CCC weight based on current epoch."""
        self.ccc_weight = min(epoch / max(self.warmup_epochs, 1), 1.0) * self.ccc_max_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(predictions, targets)
        # Normalize MSE to ~[0,1] scale
        mse_normalized = mse / (24.0 ** 2)
        ccc = self.ccc_loss(predictions, targets)
        return mse_normalized + self.ccc_weight * ccc
