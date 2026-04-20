"""
[LAYER_START] Session 10: SQLAlchemy ORM Models
Tables: predictions, experiments, model_versions, training_curves.

[INFERENCE_PATH] predictions table — stores every API prediction.
[TRAINING_PATH] experiments + training_curves — stores training run metadata.
"""

import logging
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


class Prediction(Base):
    """
    [INFERENCE_PATH] Stores every prediction made through the API.
    Enables audit trail, drift detection, and usage analytics.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    participant_id = Column(String(255), nullable=False, index=True)
    phq8_score = Column(Float, nullable=False)
    severity = Column(String(50), nullable=False)
    num_chunks = Column(Integer, nullable=False)
    inference_time_ms = Column(Float, nullable=True)
    device = Column(String(20), nullable=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)

    model_version = relationship("ModelVersion", back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, participant={self.participant_id}, "
            f"phq8={self.phq8_score:.1f}, severity={self.severity})>"
        )


class Experiment(Base):
    """
    [TRAINING_PATH] Stores metadata for each training run.
    Links to model versions and per-epoch training curves.
    """

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    config_snapshot = Column(Text, nullable=True)  # JSON string of full config
    train_samples = Column(Integer, nullable=True)
    dev_samples = Column(Integer, nullable=True)
    max_epochs = Column(Integer, nullable=True)
    actual_epochs = Column(Integer, nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    best_epoch = Column(Integer, nullable=True)
    best_ccc = Column(Float, nullable=True)
    best_rmse = Column(Float, nullable=True)
    best_mae = Column(Float, nullable=True)
    final_lr = Column(Float, nullable=True)
    stopped_reason = Column(String(50), nullable=True)  # early_stopping | max_epochs

    model_versions = relationship("ModelVersion", back_populates="experiment")
    training_curves = relationship(
        "TrainingCurve", back_populates="experiment", order_by="TrainingCurve.epoch"
    )

    def __repr__(self) -> str:
        return (
            f"<Experiment(id={self.id}, epochs={self.actual_epochs}, "
            f"best_ccc={self.best_ccc})>"
        )


class ModelVersion(Base):
    """
    Tracks deployed model versions.
    Links an experiment's best checkpoint to production use.
    """

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    version = Column(String(50), nullable=False, unique=True, index=True)
    checkpoint_path = Column(String(500), nullable=False)
    normalizer_path = Column(String(500), nullable=True)
    pca_path = Column(String(500), nullable=True)
    model_params = Column(Integer, nullable=True)
    is_production = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)

    experiment = relationship("Experiment", back_populates="model_versions")
    predictions = relationship("Prediction", back_populates="model_version")

    def __repr__(self) -> str:
        return (
            f"<ModelVersion(id={self.id}, version={self.version}, "
            f"prod={self.is_production})>"
        )


class TrainingCurve(Base):
    """
    [TRAINING_PATH] Per-epoch metrics for a training run.
    Enables loss curve visualization and early stopping analysis.
    """

    __tablename__ = "training_curves"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(
        Integer, ForeignKey("experiments.id"), nullable=False, index=True
    )
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=True)
    val_ccc = Column(Float, nullable=True)
    val_rmse = Column(Float, nullable=True)
    val_mae = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)

    experiment = relationship("Experiment", back_populates="training_curves")

    def __repr__(self) -> str:
        return (
            f"<TrainingCurve(exp={self.experiment_id}, epoch={self.epoch}, "
            f"ccc={self.val_ccc})>"
        )
