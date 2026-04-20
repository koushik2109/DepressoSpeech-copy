"""
Fusion Inference Pipeline — DepressoSpeech

End-to-end: raw audio → preprocess → extract features → fusion predict → PHQ-8 score.

Uses MultimodalFusion model which combines:
    - Text (SBERT embeddings from Whisper transcription)
    - Audio (MFCC + eGeMAPS per segment)
    - Behavioral (interview-level turn/pause/speaking-rate features from transcript)
"""

import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features import EgemapsExtractor, MfccExtractor, TextExtractor
from src.inference.fusion_predictor import FusionPredictor

logger = logging.getLogger(__name__)


@dataclass
class FusionPredictionResult:
    phq8_score: float
    severity: str
    num_chunks: int
    participant_id: str = "unknown"
    inference_time_s: float = 0.0

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


def extract_behavioral_from_chunks(chunks, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract 16 behavioral features from audio chunks at inference time.

    All features are NORMALIZED (ratios/rates, not absolute durations)
    to prevent longer recordings from receiving artificially higher scores.
    Features that need transcripts (word count, speaking rate) are approximated.

    Returns: (16,) float32
    """
    n_chunks = len(chunks)
    if n_chunks < 2:
        return np.zeros(16, dtype=np.float32)

    # Estimate durations from audio chunk lengths
    durations = np.array([len(c.audio) / sample_rate for c in chunks])
    if hasattr(chunks[0], 'start_time') and chunks[0].start_time is not None:
        starts = np.array([c.start_time for c in chunks])
        ends = np.array([c.end_time for c in chunks])
        gaps = starts[1:] - ends[:-1]
        gaps = np.maximum(gaps, 0)
    else:
        gaps = np.full(n_chunks - 1, 0.5, dtype=np.float32)

    total_speak = durations.sum()
    total_interview = total_speak + gaps.sum()
    speaking_ratio = total_speak / max(total_interview, 1.0)

    # Word count approximation: ~2.5 words per second of speech
    word_counts = (durations * 2.5).astype(int)
    speaking_rates = word_counts / np.maximum(durations, 0.1)

    # NOTE: All absolute-duration features (total_speaking_time,
    # interview_duration, total_word_count, n_chunks) are replaced with
    # ratios/rates to prevent audio-length bias.
    return np.array([
        min(n_chunks / 10.0, 1.0),               # turn_density (0-1, capped at 10 turns)
        durations.mean(),                         # avg_turn_duration
        durations.std(),                          # turn_duration_std
        gaps.mean() if len(gaps) > 0 else 0,      # avg_pause
        gaps.std() if len(gaps) > 0 else 0,       # pause_std
        np.median(gaps) if len(gaps) > 0 else 0,  # median_pause
        (gaps > 3.0).mean() if len(gaps) > 0 else 0,  # long_pause_frac
        speaking_rates.mean(),                    # avg_speaking_rate
        speaking_rates.std(),                     # rate_std
        speaking_ratio,                           # speaking_ratio (normalized)
        durations.max() / max(durations.mean(), 0.1) - 1.0,  # turn_duration_range (ratio)
        word_counts.mean(),                       # avg_words_per_turn
        word_counts.std(),                        # words_per_turn_std
        (durations < durations.mean()).mean(),     # short_turn_frac
        (durations > durations.mean()).mean(),     # long_turn_frac
        (gaps > gaps.mean()).mean() if len(gaps) > 0 else 0,  # above_avg_pause_frac
    ], dtype=np.float32)


class FusionInferencePipeline:
    """
    End-to-end inference pipeline for the fusion model.

    Process:
        1. AudioPreprocessor: load → resample → VAD → chunk
        2. EgemapsExtractor: chunks → eGeMAPS (N, 88)
        3. MfccExtractor: chunks → MFCC (N, 120)
        4. TextExtractor: chunks → Whisper → SBERT (N, 384)
        5. Behavioral: extract from chunk timing
        6. FusionPredictor: text + audio + behavioral → PHQ-8
    """

    def __init__(
        self,
        fusion_checkpoint: str = "checkpoints/best_fusion.pt",
        text_checkpoint: str = "checkpoints/best_model.pt",
        audio_config: Optional[dict] = None,
        device: str = "auto",
    ):
        self.fusion_checkpoint = fusion_checkpoint
        self.text_checkpoint = text_checkpoint
        self.audio_config = audio_config
        self.device = device

        # Lazy initialization
        self._preprocessor = None
        self._egemaps = None
        self._mfcc = None
        self._text = None
        self._predictor = None
        self.device = device

        logger.info(
            f"FusionInferencePipeline initialized: "
            f"fusion={Path(fusion_checkpoint).name}, text={Path(text_checkpoint).name}"
        )

    def _get_preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = AudioPreprocessor(config=self.audio_config)
        return self._preprocessor

    def _get_egemaps(self):
        if self._egemaps is None:
            self._egemaps = EgemapsExtractor()
        return self._egemaps

    def _get_mfcc(self):
        if self._mfcc is None:
            self._mfcc = MfccExtractor()
        return self._mfcc

    def _get_text(self):
        if self._text is None:
            self._text = TextExtractor()
        return self._text

    def _get_predictor(self):
        if self._predictor is None:
            self._predictor = FusionPredictor(
                fusion_checkpoint=self.fusion_checkpoint,
                text_checkpoint=self.text_checkpoint,
                device=self.device,
            )
        return self._predictor

    def predict_from_audio(
        self, audio_path: str, participant_id: str = "unknown"
    ) -> FusionPredictionResult:
        """
        Full pipeline: audio file → PHQ-8 score.

        Args:
            audio_path: Path to audio file (wav, mp3, flac, etc.)
            participant_id: Identifier for logging

        Returns:
            FusionPredictionResult with PHQ-8 score, severity, etc.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t_start = time.perf_counter()

        # 1. Preprocess: load → resample → VAD → chunk
        preprocessor = self._get_preprocessor()
        chunk_result = preprocessor.process_single(
            audio_path=str(audio_path), participant_id=participant_id,
        )

        if chunk_result.num_chunks == 0:
            raise ValueError(f"No usable audio chunks for '{participant_id}'")

        audio_chunks = np.stack([c.audio for c in chunk_result.chunks])
        sample_rate = preprocessor.sample_rate

        # 2. Extract per-chunk features
        egemaps = self._get_egemaps().extract_from_audio(audio_chunks, sample_rate)
        mfcc = self._get_mfcc().extract_from_audio(audio_chunks, sample_rate)
        text_emb = self._get_text().extract_from_audio(audio_chunks, sample_rate)

        # Align chunk counts
        min_chunks = min(egemaps.shape[0], mfcc.shape[0], text_emb.shape[0])
        egemaps = egemaps[:min_chunks]
        mfcc = mfcc[:min_chunks]
        text_emb = text_emb[:min_chunks]

        # 3. Extract behavioral features
        behavioral = extract_behavioral_from_chunks(
            chunk_result.chunks[:min_chunks], sample_rate
        )

        # 4. Predict
        predictor = self._get_predictor()
        score = predictor.predict(text_emb, mfcc, egemaps, behavioral)
        display_score = max(0.0, min(24.0, score))

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Prediction: pid={participant_id}, score={display_score:.2f}, "
            f"chunks={min_chunks}, time={elapsed:.2f}s"
        )

        return FusionPredictionResult(
            phq8_score=round(display_score, 2),
            severity=FusionPredictionResult.severity_label(display_score),
            num_chunks=min_chunks,
            participant_id=participant_id,
            inference_time_s=round(elapsed, 3),
        )

    def predict_from_features(
        self,
        text_features: np.ndarray,
        mfcc_features: np.ndarray,
        egemaps_features: np.ndarray,
        behavioral: Optional[np.ndarray] = None,
        participant_id: str = "unknown",
    ) -> FusionPredictionResult:
        """
        Predict from pre-extracted features (for batch processing or testing).
        """
        predictor = self._get_predictor()
        score = predictor.predict(text_features, mfcc_features, egemaps_features, behavioral)
        display_score = max(0.0, min(24.0, score))

        return FusionPredictionResult(
            phq8_score=round(display_score, 2),
            severity=FusionPredictionResult.severity_label(display_score),
            num_chunks=text_features.shape[0],
            participant_id=participant_id,
        )

    def predict_batch(self, audio_paths, participant_ids=None):
        """Run inference on multiple audio files."""
        if participant_ids is None:
            participant_ids = [Path(p).stem for p in audio_paths]

        results = []
        for audio_path, pid in zip(audio_paths, participant_ids):
            try:
                result = self.predict_from_audio(str(audio_path), pid)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed: {pid}: {e}")
        return results
