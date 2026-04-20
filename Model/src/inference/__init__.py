"""
Inference Pipeline — DepressoSpeech
Text-only: raw audio → PHQ-8 prediction (legacy)
Fusion: raw audio → multimodal fusion → PHQ-8 prediction
"""
from .predictor import Predictor
from .pipeline import InferencePipeline, PredictionResult
from .fusion_predictor import FusionPredictor
from .fusion_pipeline import FusionInferencePipeline, FusionPredictionResult

__all__ = [
    "Predictor",
    "InferencePipeline",
    "PredictionResult",
    "FusionPredictor",
    "FusionInferencePipeline",
    "FusionPredictionResult",
]
