"""
[LAYER_START] Session 8: Inference Pipeline
End-to-end: raw audio → preprocess → extract features → normalize → fuse → PCA → predict.

[INFERENCE_PATH] Chains all Session 2-8 components with saved training artifacts.
Deterministic output: no augmentation, no dropout, no randomness.
"""

import numpy as np
import torch
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features import (
    EgemapsExtractor,
    MfccExtractor,
    TextExtractor,
    FeatureNormalizer,
    FeatureFusion,
    PCAReducer,
)
from src.inference.predictor import Predictor
from src.features.audio_quality import AudioQualityScorer

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for a single inference result."""
    phq8_score: float
    severity: str
    num_chunks: int
    participant_id: str = "unknown"

    @staticmethod
    def severity_label(score: float) -> str:
        if score < 5:
            return "none/minimal"
        elif score < 10:
            return "mild"
        elif score < 15:
            return "moderate"
        elif score < 20:
            return "moderately severe"
        else:
            return "severe"


@dataclass
class ExtendedPredictionResult:
    """Extended inference result with confidence, audio quality, and behavioral features."""
    phq8_score: float
    severity: str
    num_chunks: int
    participant_id: str
    inference_time_s: float
    confidence: dict       # {mean, std, ci_lower, ci_upper}
    audio_quality: dict    # {rms, snr_db, speech_prob, quality}
    behavioral: dict       # eGeMAPS-derived behavioral features


class InferencePipeline:
    """
    End-to-end inference pipeline: raw audio file → PHQ-8 prediction.

    Chains all components with exact training parity:
        1. AudioPreprocessor: load → resample → VAD → chunk
        2. EgemapsExtractor: chunks → eGeMAPS (N, 88)
        3. MfccExtractor: chunks → MFCC (N, 120)
        4. TextExtractor: chunks → Whisper → SBERT (N, 384)
        5. FeatureNormalizer: load saved scalers → normalize
        6. FeatureFusion: concatenate → (N, 592)
        7. PCAReducer: load saved PCA → reduce → (N, 64)
        8. Predictor: load saved model → predict → PHQ-8 score

    All components are lazy-loaded on first use to minimize startup time.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        normalizer_path: Union[str, Path, None] = None,
        pca_path: Union[str, Path, None] = None,
        audio_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
        device: str = "auto",
    ):
        """
        Args:
            model_path: Path to trained model checkpoint (.pt)
            normalizer_path: Path to saved normalizer (.pkl). Defaults to standard path.
            pca_path: Path to saved PCA (.pkl). Defaults to standard path.
            audio_config: Audio preprocessing config dict. None uses defaults.
            model_config: Model architecture config. None loads from checkpoint.
            device: "auto", "cpu", or "cuda"
        """
        self.model_path = Path(model_path)
        self.normalizer_path = normalizer_path
        self.pca_path = pca_path
        self.audio_config = audio_config
        self.model_config = model_config
        self.device = device

        # Lazy-loaded components
        self._preprocessor: Optional[AudioPreprocessor] = None
        self._egemaps: Optional[EgemapsExtractor] = None
        self._mfcc: Optional[MfccExtractor] = None
        self._text: Optional[TextExtractor] = None
        self._normalizer: Optional[FeatureNormalizer] = None
        self._fusion: Optional[FeatureFusion] = None
        self._pca: Optional[PCAReducer] = None
        self._predictor: Optional[Predictor] = None

        logger.info(
            f"[INFERENCE_PATH] Pipeline initialized: model={self.model_path.name}"
        )

    # =========================================================
    # Lazy component initialization
    # =========================================================
    def _get_preprocessor(self) -> AudioPreprocessor:
        if self._preprocessor is None:
            self._preprocessor = AudioPreprocessor(config=self.audio_config)
            logger.info("[INFERENCE_PATH] AudioPreprocessor initialized")
        return self._preprocessor

    def _get_egemaps(self) -> EgemapsExtractor:
        if self._egemaps is None:
            self._egemaps = EgemapsExtractor()
        return self._egemaps

    def _get_mfcc(self) -> MfccExtractor:
        if self._mfcc is None:
            self._mfcc = MfccExtractor()
        return self._mfcc

    def _get_text(self) -> TextExtractor:
        if self._text is None:
            self._text = TextExtractor()
        return self._text

    def _get_normalizer(self) -> FeatureNormalizer:
        if self._normalizer is None:
            self._normalizer = FeatureNormalizer()
            self._normalizer.load(self.normalizer_path)
            logger.info("[INFERENCE_PATH] FeatureNormalizer loaded")
        return self._normalizer

    def _get_fusion(self) -> FeatureFusion:
        if self._fusion is None:
            self._fusion = FeatureFusion()
        return self._fusion

    def _get_pca(self) -> PCAReducer:
        if self._pca is None:
            self._pca = PCAReducer()
            self._pca.load(self.pca_path)
            logger.info("[INFERENCE_PATH] PCAReducer loaded")
        return self._pca

    def _get_predictor(self) -> Predictor:
        if self._predictor is None:
            self._predictor = Predictor(
                model_path=self.model_path,
                model_config=self.model_config,
                device=self.device,
            )
        return self._predictor

    # =========================================================
    # Core inference
    # =========================================================
    def predict_from_audio(
        self,
        audio_path: Union[str, Path],
        participant_id: str = "unknown",
    ) -> PredictionResult:
        """
        End-to-end inference from raw audio file.

        Pipeline:
            audio → preprocess → extract (egemaps + mfcc + text) →
            normalize → fuse (592) → PCA (64) → model → PHQ-8

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            participant_id: Optional identifier for logging

        Returns:
            PredictionResult with PHQ-8 score and severity label
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"[INFERENCE_PATH] Processing: {audio_path.name} ({participant_id})")
        t_start = time.perf_counter()

        # Step 1: Preprocess audio → chunks
        t0 = time.perf_counter()
        preprocessor = self._get_preprocessor()
        chunk_result = preprocessor.process_single(
            audio_path=str(audio_path),
            participant_id=participant_id,
        )
        logger.debug(f"[TIMING] preprocess: {time.perf_counter() - t0:.3f}s")

        if chunk_result.num_chunks == 0:
            logger.warning(
                f"[VALIDATION_CHECK] {participant_id}: No audio chunks after preprocessing"
            )
            raise ValueError(
                f"No usable audio chunks after preprocessing for participant '{participant_id}'"
            )

        # Convert List[AudioChunk] → np.ndarray (num_chunks, samples_per_chunk)
        audio_chunks = np.stack([c.audio for c in chunk_result.chunks])

        # Step 2: Extract features from chunks
        t0 = time.perf_counter()
        egemaps = self._get_egemaps().extract_from_audio(audio_chunks, preprocessor.sample_rate)
        mfcc = self._get_mfcc().extract_from_audio(audio_chunks, preprocessor.sample_rate)
        text_emb = self._get_text().extract_from_audio(audio_chunks, preprocessor.sample_rate)
        logger.debug(f"[TIMING] feature_extraction: {time.perf_counter() - t0:.3f}s")

        # Align chunk counts (defensive)
        min_chunks = min(egemaps.shape[0], mfcc.shape[0], text_emb.shape[0])
        egemaps = egemaps[:min_chunks]
        mfcc = mfcc[:min_chunks]
        text_emb = text_emb[:min_chunks]

        logger.info(
            f"[DATA_FLOW] {participant_id}: {min_chunks} chunks, "
            f"egemaps={egemaps.shape}, mfcc={mfcc.shape}, text={text_emb.shape}"
        )

        # Steps 3-7: Normalize → Fuse → PCA → Predict
        result = self._predict_from_features(egemaps, mfcc, text_emb, participant_id)
        logger.info(f"[TIMING] total_inference: {time.perf_counter() - t_start:.3f}s ({participant_id})")
        return result

    def predict_from_features(
        self,
        egemaps: np.ndarray,
        mfcc: np.ndarray,
        text_embeddings: np.ndarray,
        participant_id: str = "unknown",
    ) -> PredictionResult:
        """
        Inference from pre-extracted raw (unnormalized) features.

        Runs the full normalize → fuse → PCA → predict pipeline.
        Useful when features are already extracted (e.g., from CSV for
        evaluation) but have NOT been normalized or fused yet.

        Args:
            egemaps: (N, 88) raw (unnormalized) eGeMAPS features
            mfcc: (N, 120) raw (unnormalized) MFCC features
            text_embeddings: (N, 384) raw text embeddings
            participant_id: Optional identifier

        Returns:
            PredictionResult
        """
        return self._predict_from_features(egemaps, mfcc, text_embeddings, participant_id)

    def _predict_from_features(
        self,
        egemaps: np.ndarray,
        mfcc: np.ndarray,
        text_embeddings: np.ndarray,
        participant_id: str,
    ) -> PredictionResult:
        """
        Internal: normalize → fuse → PCA → predict.

        Args:
            egemaps: (N, 88)
            mfcc: (N, 120)
            text_embeddings: (N, 384)
            participant_id: identifier for logging

        Returns:
            PredictionResult
        """
        num_chunks = egemaps.shape[0]

        # Step 3: Normalize
        t0 = time.perf_counter()
        normalizer = self._get_normalizer()
        normed = normalizer.transform(egemaps, mfcc, text_embeddings)

        # Step 4: Fuse → (N, 592)
        fusion = self._get_fusion()
        fused = fusion.fuse(normed)

        # Step 5: PCA reduce → (N, 64)
        pca = self._get_pca()
        reduced = pca.transform(fused)
        logger.debug(f"[TIMING] normalize+fuse+pca: {time.perf_counter() - t0:.3f}s")

        # Step 6: Predict
        t0 = time.perf_counter()
        predictor = self._get_predictor()
        features_t = torch.from_numpy(reduced).float()  # BUG-4: Ensure float32
        score = predictor.predict_single(features_t)
        logger.debug(f"[TIMING] model_predict: {time.perf_counter() - t0:.3f}s")

        # Clamp to valid PHQ-8 range [0, 24]
        score = max(0.0, min(24.0, score))

        result = PredictionResult(
            phq8_score=round(score, 2),
            severity=PredictionResult.severity_label(score),
            num_chunks=num_chunks,
            participant_id=participant_id,
        )

        logger.info(
            f"[INFERENCE_PATH] {participant_id}: PHQ-8={result.phq8_score} "
            f"({result.severity}), chunks={num_chunks}"
        )
        return result

    def predict_from_audio_extended(
        self,
        audio_path: Union[str, Path],
        participant_id: str = "unknown",
    ) -> "ExtendedPredictionResult":
        """
        Extended inference: audio → PHQ-8 + confidence intervals + audio quality + behavioral features.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t_start = time.perf_counter()

        # Step 1: Preprocess audio → chunks
        preprocessor = self._get_preprocessor()
        chunk_result = preprocessor.process_single(
            audio_path=str(audio_path),
            participant_id=participant_id,
        )

        if chunk_result.num_chunks == 0:
            raise ValueError(f"No usable audio chunks for '{participant_id}'")

        audio_chunks = np.stack([c.audio for c in chunk_result.chunks])

        # Audio quality: score the concatenated raw audio
        quality_scorer = AudioQualityScorer()
        concat_audio = np.concatenate([c.audio for c in chunk_result.chunks])
        quality_score, quality_details = quality_scorer.score_segment(concat_audio)

        # Step 2: Extract features
        egemaps = self._get_egemaps().extract_from_audio(audio_chunks, preprocessor.sample_rate)
        mfcc = self._get_mfcc().extract_from_audio(audio_chunks, preprocessor.sample_rate)
        text_emb = self._get_text().extract_from_audio(audio_chunks, preprocessor.sample_rate)

        min_chunks = min(egemaps.shape[0], mfcc.shape[0], text_emb.shape[0])
        egemaps = egemaps[:min_chunks]
        mfcc = mfcc[:min_chunks]
        text_emb = text_emb[:min_chunks]

        # Behavioral features from eGeMAPS means (88 features per chunk → aggregate)
        egemaps_mean = egemaps.mean(axis=0)  # (88,)
        behavioral = {
            "f0_mean": float(egemaps_mean[0]) if egemaps_mean.shape[0] > 0 else 0.0,
            "f0_std": float(egemaps_mean[1]) if egemaps_mean.shape[0] > 1 else 0.0,
            "jitter": float(egemaps_mean[2]) if egemaps_mean.shape[0] > 2 else 0.0,
            "shimmer": float(egemaps_mean[3]) if egemaps_mean.shape[0] > 3 else 0.0,
            "loudness_mean": float(egemaps_mean[4]) if egemaps_mean.shape[0] > 4 else 0.0,
            "loudness_std": float(egemaps_mean[5]) if egemaps_mean.shape[0] > 5 else 0.0,
            "speaking_rate": float(min_chunks / max((time.perf_counter() - t_start), 0.01)),
            "num_chunks": min_chunks,
            "avg_chunk_duration": float(concat_audio.shape[0] / preprocessor.sample_rate / max(min_chunks, 1)),
            "total_duration": float(concat_audio.shape[0] / preprocessor.sample_rate),
        }

        # Normalize → Fuse → PCA → Predict with uncertainty
        normalizer = self._get_normalizer()
        normed = normalizer.transform(egemaps, mfcc, text_emb)
        fused = self._get_fusion().fuse(normed)
        reduced = self._get_pca().transform(fused)

        features_t = torch.from_numpy(reduced).float()
        predictor = self._get_predictor()

        # MC Dropout confidence intervals
        try:
            confidence = predictor.predict_with_uncertainty(features_t, n_samples=30)
        except Exception:
            # Fallback to simple prediction if MC dropout fails
            score = predictor.predict_single(features_t)
            confidence = {"mean": score, "std": 0.0, "ci_lower": score, "ci_upper": score}

        score = max(0.0, min(24.0, confidence["mean"]))
        inference_time = time.perf_counter() - t_start

        return ExtendedPredictionResult(
            phq8_score=round(score, 2),
            severity=PredictionResult.severity_label(score),
            num_chunks=min_chunks,
            participant_id=participant_id,
            inference_time_s=round(inference_time, 3),
            confidence=confidence,
            audio_quality={
                "rms": round(quality_details["rms"], 4),
                "snr_db": round(quality_details["snr_db"], 2),
                "speech_prob": round(quality_details["speech_prob"], 3),
                "quality": round(quality_score, 3),
            },
            behavioral=behavioral,
        )

    def predict_batch(
        self,
        audio_paths: list,
        participant_ids: Optional[list] = None,
    ) -> list:
        """
        Batch inference for multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            participant_ids: Optional list of identifiers (same length as audio_paths)

        Returns:
            List of PredictionResult
        """
        if participant_ids is None:
            participant_ids = [Path(p).stem for p in audio_paths]

        results = []
        for audio_path, pid in zip(audio_paths, participant_ids):
            try:
                result = self.predict_from_audio(audio_path, pid)
                results.append(result)
            except Exception as e:
                logger.error(f"[DEBUG_POINT] {pid} failed: {e}")
                continue

        return results
