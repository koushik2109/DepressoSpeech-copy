"""
[LAYER_START] Session 9: API Schemas
Request/response models for the REST API.

[INFERENCE_PATH] HTTP request → validated schema → inference pipeline → response schema.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# =========================================================
# Response Schemas
# =========================================================


class PredictionResponse(BaseModel):
    """Single prediction result returned by the API."""

    participant_id: str = Field(..., description="Identifier for the audio sample")
    phq8_score: float = Field(..., ge=0.0, le=24.0, description="Predicted PHQ-8 score (0-24)")
    severity: str = Field(..., description="Clinical severity category")
    num_chunks: int = Field(..., ge=0, description="Number of audio chunks processed")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 UTC timestamp of prediction",
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction endpoint."""

    predictions: List[PredictionResponse]
    total: int = Field(..., description="Total number of files processed")
    failed: List[str] = Field(
        default_factory=list, description="Files that failed processing"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Inference device (cpu/cuda)")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Human-readable error message")


class ExtendedPredictionResponse(BaseModel):
    """Extended prediction with confidence, audio quality, and behavioral features."""

    participant_id: str = Field(..., description="Identifier for the audio sample")
    phq8_score: float = Field(..., ge=0.0, le=24.0, description="Predicted PHQ-8 score")
    severity: str = Field(..., description="Clinical severity category")
    num_chunks: int = Field(..., ge=0, description="Number of audio chunks processed")
    inference_time_s: float = Field(..., description="Total inference time in seconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )
    confidence: dict = Field(..., description="MC Dropout uncertainty: mean, std, ci_lower, ci_upper")
    audio_quality: dict = Field(..., description="Audio quality metrics: rms, snr_db, speech_prob, quality")
    behavioral: dict = Field(..., description="Behavioral features extracted from audio")
