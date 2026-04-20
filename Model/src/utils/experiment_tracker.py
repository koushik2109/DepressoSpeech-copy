"""
[LAYER_START] Session 10: Experiment Tracker
High-level API for logging training runs and predictions to the database.

Decouples the trainer/API from raw SQLAlchemy operations.

Usage (training):
    tracker = ExperimentTracker()
    exp_id = tracker.start_experiment(config, train_samples=163, dev_samples=56)
    tracker.log_epoch(exp_id, epoch=1, train_loss=0.5, val_ccc=0.3, ...)
    tracker.finish_experiment(exp_id, best_epoch=42, best_ccc=0.45, ...)

Usage (inference):
    tracker = ExperimentTracker()
    tracker.log_prediction(participant_id="P001", phq8_score=12.5, ...)
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, sessionmaker

from src.db.database import init_db
from src.db.models import Experiment, ModelVersion, Prediction, TrainingCurve

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    High-level interface for database logging.

    Wraps SQLAlchemy session management and provides clean methods
    for training and inference logging.
    """

    def __init__(
        self,
        session_factory: Optional[sessionmaker] = None,
        db_config_path: str = "configs/db_config.yaml",
    ):
        """
        Initialize tracker.

        Args:
            session_factory: Pre-built sessionmaker. If None, initializes
                           from db_config.yaml (creates tables if needed).
            db_config_path: Path to db_config.yaml (used if session_factory is None).
        """
        if session_factory is not None:
            self._factory = session_factory
        else:
            self._factory = init_db(db_config_path)

    def _get_session(self) -> Session:
        """Create a new session."""
        return self._factory()

    # =========================================================
    # Training: Experiment lifecycle
    # =========================================================

    def start_experiment(
        self,
        config: Dict[str, Any],
        train_samples: int,
        dev_samples: int,
        max_epochs: int,
    ) -> int:
        """
        Record the start of a training run.

        Args:
            config: Full training config dict (saved as JSON snapshot).
            train_samples: Number of training samples.
            dev_samples: Number of dev/validation samples.
            max_epochs: Maximum epochs configured.

        Returns:
            experiment_id for use in subsequent calls.
        """
        session = self._get_session()
        try:
            exp = Experiment(
                config_snapshot=json.dumps(config, default=str),
                train_samples=train_samples,
                dev_samples=dev_samples,
                max_epochs=max_epochs,
            )
            session.add(exp)
            session.commit()
            exp_id = exp.id
            logger.info(f"[DB] Experiment started: id={exp_id}")
            return exp_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def log_epoch(
        self,
        experiment_id: int,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_ccc: Optional[float] = None,
        val_rmse: Optional[float] = None,
        val_mae: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Log metrics for a single training epoch."""
        session = self._get_session()
        try:
            curve = TrainingCurve(
                experiment_id=experiment_id,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_ccc=val_ccc,
                val_rmse=val_rmse,
                val_mae=val_mae,
                learning_rate=learning_rate,
            )
            session.add(curve)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def finish_experiment(
        self,
        experiment_id: int,
        actual_epochs: int,
        training_time_seconds: float,
        best_epoch: int,
        best_ccc: float,
        best_rmse: float,
        best_mae: float,
        final_lr: float,
        stopped_reason: str = "early_stopping",
    ) -> None:
        """
        Record final results of a completed training run.

        Args:
            experiment_id: ID from start_experiment().
            actual_epochs: Epochs actually run (≤ max_epochs).
            training_time_seconds: Total wall-clock training time.
            best_epoch: Epoch with best validation CCC.
            best_ccc: Best concordance correlation coefficient.
            best_rmse: RMSE at best epoch.
            best_mae: MAE at best epoch.
            final_lr: Learning rate at end of training.
            stopped_reason: "early_stopping" or "max_epochs".
        """
        session = self._get_session()
        try:
            exp = session.get(Experiment, experiment_id)
            if exp is None:
                logger.warning(f"[DB] Experiment {experiment_id} not found")
                return

            exp.actual_epochs = actual_epochs
            exp.training_time_seconds = round(training_time_seconds, 1)
            exp.best_epoch = best_epoch
            exp.best_ccc = round(best_ccc, 4)
            exp.best_rmse = round(best_rmse, 4)
            exp.best_mae = round(best_mae, 4)
            exp.final_lr = final_lr
            exp.stopped_reason = stopped_reason
            session.commit()
            logger.info(
                f"[DB] Experiment {experiment_id} finished: "
                f"epochs={actual_epochs}, best_ccc={best_ccc:.4f}, "
                f"reason={stopped_reason}"
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # =========================================================
    # Inference: Prediction logging
    # =========================================================

    def log_prediction(
        self,
        participant_id: str,
        phq8_score: float,
        severity: str,
        num_chunks: int,
        inference_time_ms: Optional[float] = None,
        device: Optional[str] = None,
        model_version_id: Optional[int] = None,
    ) -> int:
        """
        Log a single prediction to the database.

        Returns:
            prediction_id
        """
        session = self._get_session()
        try:
            pred = Prediction(
                participant_id=participant_id,
                phq8_score=round(phq8_score, 2),
                severity=severity,
                num_chunks=num_chunks,
                inference_time_ms=round(inference_time_ms, 1) if inference_time_ms else None,
                device=device,
                model_version_id=model_version_id,
            )
            session.add(pred)
            session.commit()
            pred_id = pred.id
            logger.debug(
                f"[DB] Prediction logged: id={pred_id}, "
                f"participant={participant_id}, phq8={phq8_score:.1f}"
            )
            return pred_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def log_predictions_batch(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Log multiple predictions in a single transaction.

        Args:
            predictions: List of dicts with keys matching log_prediction params.

        Returns:
            List of prediction IDs.
        """
        session = self._get_session()
        try:
            records = []
            for p in predictions:
                pred = Prediction(
                    participant_id=p["participant_id"],
                    phq8_score=round(p["phq8_score"], 2),
                    severity=p["severity"],
                    num_chunks=p["num_chunks"],
                    inference_time_ms=(
                        round(p["inference_time_ms"], 1)
                        if p.get("inference_time_ms")
                        else None
                    ),
                    device=p.get("device"),
                    model_version_id=p.get("model_version_id"),
                )
                records.append(pred)
            session.add_all(records)
            session.commit()
            ids = [r.id for r in records]
            logger.debug(f"[DB] Batch logged: {len(ids)} predictions")
            return ids
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # =========================================================
    # Model version management
    # =========================================================

    def register_model_version(
        self,
        version: str,
        checkpoint_path: str,
        experiment_id: Optional[int] = None,
        normalizer_path: Optional[str] = None,
        pca_path: Optional[str] = None,
        model_params: Optional[int] = None,
        is_production: bool = False,
        notes: Optional[str] = None,
    ) -> int:
        """
        Register a new model version.

        Returns:
            model_version_id
        """
        session = self._get_session()
        try:
            mv = ModelVersion(
                version=version,
                checkpoint_path=checkpoint_path,
                experiment_id=experiment_id,
                normalizer_path=normalizer_path,
                pca_path=pca_path,
                model_params=model_params,
                is_production=is_production,
                notes=notes,
            )
            session.add(mv)
            session.commit()
            mv_id = mv.id
            logger.info(f"[DB] Model version registered: {version} (id={mv_id})")
            return mv_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_production_model(self) -> Optional[ModelVersion]:
        """Get the current production model version, if any."""
        session = self._get_session()
        try:
            return (
                session.query(ModelVersion)
                .filter(ModelVersion.is_production == True)
                .order_by(ModelVersion.created_at.desc())
                .first()
            )
        finally:
            session.close()

    # =========================================================
    # Query helpers
    # =========================================================

    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Fetch an experiment by ID."""
        session = self._get_session()
        try:
            return session.get(Experiment, experiment_id)
        finally:
            session.close()

    def get_recent_predictions(self, limit: int = 50) -> List[Prediction]:
        """Fetch the most recent predictions."""
        session = self._get_session()
        try:
            return (
                session.query(Prediction)
                .order_by(Prediction.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def get_prediction_count(self) -> int:
        """Total number of predictions in the database."""
        session = self._get_session()
        try:
            return session.query(Prediction).count()
        finally:
            session.close()
