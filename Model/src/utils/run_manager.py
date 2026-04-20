"""
[LAYER_START] Session 10: Run Manager
Organizes checkpoints and logs with clear, descriptive naming.

Checkpoint naming convention:
    checkpoints/
    ├── best_model.pt                 ← best validation CCC checkpoint
    ├── latest_model.pt               ← most recent epoch checkpoint
    └── scalers/
        ├── feature_scalers.pkl       ← fitted FeatureNormalizer
        └── pca_reducer.pkl           ← fitted PCAReducer

Log naming convention:
    logs/
    ├── training.log                  ← full training run log
    ├── feature_extraction.log        ← feature extraction log
    ├── inference.log                 ← prediction/inference log
    └── api_server.log                ← REST API server log

Each checkpoint file contains metadata for traceability:
    - model_config: architecture parameters
    - epoch: training epoch number
    - best_metric: best validation CCC
    - metrics: full validation metrics dict
    - config: full training config snapshot
"""

import logging
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class RunManager:
    """
    Manages checkpoint artifacts and log directories.

    Provides:
        - Descriptive checkpoint naming with metadata logging
        - Artifact path resolution
        - Training summary export
        - Config snapshot saving
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.scalers_dir = self.checkpoint_dir / "scalers"

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scalers_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[CHECKPOINT] RunManager: checkpoints={self.checkpoint_dir}, "
            f"logs={self.log_dir}"
        )

    # =========================================================
    # Standard artifact paths
    # =========================================================

    @property
    def best_model_path(self) -> Path:
        """Path for the best validation checkpoint."""
        return self.checkpoint_dir / "best_model.pt"

    @property
    def latest_model_path(self) -> Path:
        """Path for the latest epoch checkpoint."""
        return self.checkpoint_dir / "latest_model.pt"

    @property
    def normalizer_path(self) -> Path:
        """Path for the fitted FeatureNormalizer."""
        return self.scalers_dir / "feature_scalers.pkl"

    @property
    def pca_path(self) -> Path:
        """Path for the fitted PCAReducer."""
        return self.scalers_dir / "pca_reducer.pkl"

    @property
    def training_curves_path(self) -> Path:
        """Path for per-epoch training metrics JSON."""
        return self.log_dir / "training_curves.json"

    @property
    def training_summary_path(self) -> Path:
        """Path for final training summary JSON."""
        return self.log_dir / "training_summary.json"

    @property
    def config_snapshot_path(self) -> Path:
        """Path for config snapshot used during training."""
        return self.log_dir / "config_snapshot.json"

    # =========================================================
    # Save/load helpers
    # =========================================================

    def save_config_snapshot(self, config: Dict[str, Any]) -> Path:
        """
        Save a snapshot of all configs used for this run.

        This allows exact reproduction of the training setup.
        """
        path = self.config_snapshot_path
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"[CHECKPOINT] Config snapshot saved: {path}")
        return path

    def save_training_summary(
        self,
        best_metrics: Dict[str, float],
        total_epochs: int,
        training_time_seconds: float,
        model_params: int,
        train_samples: int,
        dev_samples: int,
    ) -> Path:
        """
        Save a human-readable training summary with all key numbers.

        This is the single file to check after a training run.
        """
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "best_metrics": {k: round(v, 4) for k, v in best_metrics.items()},
            "total_epochs": total_epochs,
            "training_time_seconds": round(training_time_seconds, 1),
            "training_time_human": _format_duration(training_time_seconds),
            "model_params": model_params,
            "train_samples": train_samples,
            "dev_samples": dev_samples,
            "param_to_sample_ratio": round(model_params / max(train_samples, 1), 1),
            "artifacts": {
                "best_model": str(self.best_model_path),
                "normalizer": str(self.normalizer_path),
                "pca_reducer": str(self.pca_path),
                "training_curves": str(self.training_curves_path),
            },
        }

        path = self.training_summary_path
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"[CHECKPOINT] Training summary saved: {path}")
        return path

    def log_artifact_saved(self, name: str, path: Path, **metadata) -> None:
        """Log that an artifact was saved, with optional metadata."""
        parts = [f"[CHECKPOINT] Saved {name}: {path}"]
        for k, v in metadata.items():
            parts.append(f"{k}={v}")
        logger.info(" | ".join(parts))

    def list_artifacts(self) -> Dict[str, str]:
        """List all checkpoint artifacts and their status (exists/missing)."""
        artifacts = {
            "best_model": self.best_model_path,
            "latest_model": self.latest_model_path,
            "normalizer": self.normalizer_path,
            "pca_reducer": self.pca_path,
            "training_curves": self.training_curves_path,
            "training_summary": self.training_summary_path,
            "config_snapshot": self.config_snapshot_path,
        }

        status = {}
        for name, path in artifacts.items():
            exists = path.exists()
            status[name] = f"{'✓' if exists else '✗'} {path}"

        return status

    def print_artifact_status(self) -> None:
        """Print a formatted table of all artifacts and their status."""
        status = self.list_artifacts()
        logger.info("[CHECKPOINT] Artifact status:")
        for name, info in status.items():
            logger.info(f"  {name:<20s} {info}")


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"
