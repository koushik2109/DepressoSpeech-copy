"""
[LAYER_START] Session 7: Early Stopping
Monitors validation CCC and stops training when no improvement.

[TRAINING PATH] Tracks best metric, counts patience, triggers stop.
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping based on validation metric (CCC — higher is better).

    Stops training if no improvement of at least min_delta
    for `patience` consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max",
    ):
        """
        Args:
            patience: Epochs to wait without improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for CCC (higher=better), 'min' for loss (lower=better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

        logger.info(
            f"[LAYER_START] EarlyStopping: patience={patience}, "
            f"min_delta={min_delta}, mode={mode}"
        )

    def __call__(self, metric: float, epoch: int) -> bool:
        """
        Check whether to stop training.

        Args:
            metric: Current epoch's validation metric value
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            return False

        improved = self._is_improvement(metric)

        if improved:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"[TRAINING_PATH] Early stopping triggered at epoch {epoch}. "
                    f"Best: {self.best_score:.4f} at epoch {self.best_epoch}"
                )
                return True

        return False

    def _is_improvement(self, metric: float) -> bool:
        """Check if metric improved by at least min_delta."""
        if self.mode == "max":
            return metric > self.best_score + self.min_delta
        else:
            return metric < self.best_score - self.min_delta

    def state_dict(self) -> dict:
        """For checkpoint saving."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'should_stop': self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        """For checkpoint loading."""
        self.counter = state['counter']
        self.best_score = state['best_score']
        self.best_epoch = state['best_epoch']
        self.should_stop = state['should_stop']
