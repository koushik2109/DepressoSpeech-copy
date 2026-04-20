"""
[LAYER_START] Session 4: Feature Fusion
Concatenates normalized feature groups into a single vector per chunk/segment.

Training path: Fuse normalized train/dev/test features
Inference path: Fuse normalized inference features (identical logic)
"""

import numpy as np
import logging
from typing import Dict, Optional

from src.features.constants import EGEMAPS_DIM, MFCC_DIM, TEXT_DIM, FUSED_DIM

logger = logging.getLogger(__name__)


class FeatureFusion:
    """
    Concatenates normalized feature groups into a unified vector.

    Fusion order: [eGeMAPS (88) ⊕ MFCC (120) ⊕ Text (384)] = 592-dim
    Both training and inference use the same concatenation logic.

    [BOTH PATHS] Identical fusion for training and inference.
    """

    EXPECTED_DIMS = {
        'egemaps': EGEMAPS_DIM,
        'mfcc': MFCC_DIM,
        'text_embeddings': TEXT_DIM,
    }
    FUSED_DIM = FUSED_DIM

    def fuse(
        self,
        normalized_features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        [BOTH PATHS] Concatenate normalized features into a single vector per chunk.

        Args:
            normalized_features: Dict from FeatureNormalizer.transform() with keys:
                'egemaps' (N, 88), 'mfcc' (N, 120), 'text_embeddings' (N, 384)

        Returns:
            np.ndarray of shape (N, 592)
        """
        egemaps = normalized_features['egemaps']
        mfcc = normalized_features['mfcc']
        text = normalized_features['text_embeddings']

        # [VALIDATION_CHECK] Verify all have same number of samples
        n_samples = egemaps.shape[0]
        if mfcc.shape[0] != n_samples or text.shape[0] != n_samples:
            raise ValueError(
                f"Sample count mismatch: egemaps={egemaps.shape[0]}, "
                f"mfcc={mfcc.shape[0]}, text={text.shape[0]}. "
                f"All must have the same number of rows."
            )

        # [VALIDATION_CHECK] Verify feature dimensions
        for name, arr in [('egemaps', egemaps), ('mfcc', mfcc), ('text_embeddings', text)]:
            expected = self.EXPECTED_DIMS[name]
            if arr.shape[1] != expected:
                logger.warning(
                    f"[VALIDATION_CHECK] {name}: expected dim={expected}, "
                    f"got {arr.shape[1]}"
                )

        # [DATA_FLOW] Concatenate: [eGeMAPS | MFCC | Text] → (N, 592)
        fused = np.concatenate([egemaps, mfcc, text], axis=1).astype(np.float32)

        logger.info(
            f"[DATA_FLOW] Fused features: {fused.shape} "
            f"(egemaps={egemaps.shape[1]} + mfcc={mfcc.shape[1]} + text={text.shape[1]})"
        )

        # [VALIDATION_CHECK] Final NaN check
        nan_count = np.isnan(fused).sum()
        if nan_count > 0:
            logger.warning(
                f"[VALIDATION_CHECK] {nan_count} NaN values in fused features. "
                f"NaN→0 imputation after normalization maps to (0-mean)/std, "
                f"creating non-zero fabricated values. Review feature extraction."
            )
            fused = np.nan_to_num(fused, nan=0.0)

        return fused

    def fuse_raw(
        self,
        egemaps: np.ndarray,
        mfcc: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        [BOTH PATHS] Direct fusion from separate arrays (convenience method).

        Args:
            egemaps: (N, 88)
            mfcc: (N, 120)
            text_embeddings: (N, 384)

        Returns:
            np.ndarray of shape (N, 592)
        """
        return self.fuse({
            'egemaps': egemaps,
            'mfcc': mfcc,
            'text_embeddings': text_embeddings,
        })

    @staticmethod
    def get_feature_slices() -> Dict[str, tuple]:
        """
        Return index slices for each feature group within the fused vector.
        Useful for debugging or group-specific operations downstream.
        """
        return {
            'egemaps': (0, EGEMAPS_DIM),
            'mfcc': (EGEMAPS_DIM, EGEMAPS_DIM + MFCC_DIM),
            'text_embeddings': (EGEMAPS_DIM + MFCC_DIM, FUSED_DIM),
        }
