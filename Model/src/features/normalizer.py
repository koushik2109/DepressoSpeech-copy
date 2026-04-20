"""
[LAYER_START] Session 4: Feature Normalization
StandardScaler per feature group, with scaler persistence for inference parity.

Training path: Fit scalers on train split → transform train/dev/test
Inference path: Load saved scalers → transform new features
"""

import io
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Union

logger = logging.getLogger(__name__)

# SEC-1b: Restrict unpickling to known safe types only
_SAFE_MODULES = frozenset({
    'numpy', 'numpy.core.multiarray', 'numpy.core.numeric',
    'numpy.core._multiarray_umath', 'collections', 'builtins',
})

class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.split('.')[0] in _SAFE_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked unpickling of {module}.{name} — not in allowlist"
        )

def _safe_pickle_load(f: io.BufferedIOBase):
    return _RestrictedUnpickler(f).load()

# Default paths matching configs/normalization_config.yaml
DEFAULT_SCALER_DIR = "checkpoints/scalers"
DEFAULT_SCALER_FILENAME = "feature_scalers.pkl"
DEFAULT_SCALER_PATH = str(Path(DEFAULT_SCALER_DIR) / DEFAULT_SCALER_FILENAME)


class FeatureNormalizer:
    """
    Per-group StandardScaler for feature normalization.

    Groups:
        - egemaps: StandardScaler (fit on train)
        - mfcc: StandardScaler (fit on train)
        - text: L2 normalization (no fitting needed, pre-normalized by SBERT)

    Training path: fit() on train data → transform() on train/dev/test → save()
    Inference path: load() saved scalers → transform() new features
    """

    def __init__(self):
        self._scalers: Dict[str, Dict[str, np.ndarray]] = {}
        self._fitted = False

    # =========================================================
    # TRAINING PATH: Fit scalers on training data
    # =========================================================
    def fit(
        self,
        egemaps: np.ndarray,
        mfcc: np.ndarray,
    ) -> "FeatureNormalizer":
        """
        [TRAINING_PATH] Fit StandardScalers on training split only.

        Args:
            egemaps: Training eGeMAPS features (N, 88)
            mfcc: Training MFCC features (N, 120)

        Returns:
            self (for chaining)
        """
        # [DATA_FLOW] Compute mean and std from training data
        self._scalers['egemaps'] = {
            'mean': np.mean(egemaps, axis=0).astype(np.float32),
            'std': np.std(egemaps, axis=0).astype(np.float32),
        }
        self._scalers['mfcc'] = {
            'mean': np.mean(mfcc, axis=0).astype(np.float32),
            'std': np.std(mfcc, axis=0).astype(np.float32),
        }

        # [VALIDATION_CHECK] Prevent division by zero
        for name in ['egemaps', 'mfcc']:
            std = self._scalers[name]['std']
            zero_std = std == 0
            if zero_std.any():
                logger.warning(
                    f"[VALIDATION_CHECK] {name}: {zero_std.sum()} features have "
                    f"zero std, setting to 1.0"
                )
                self._scalers[name]['std'] = np.where(zero_std, 1.0, std)

        self._fitted = True
        logger.info(
            f"[TRAINING_PATH] Scalers fitted: "
            f"egemaps mean_range=[{self._scalers['egemaps']['mean'].min():.4f}, "
            f"{self._scalers['egemaps']['mean'].max():.4f}], "
            f"mfcc mean_range=[{self._scalers['mfcc']['mean'].min():.4f}, "
            f"{self._scalers['mfcc']['mean'].max():.4f}]"
        )
        return self

    # =========================================================
    # BOTH PATHS: Transform features using fitted/loaded scalers
    # =========================================================
    def transform(
        self,
        egemaps: np.ndarray,
        mfcc: np.ndarray,
        text_embeddings: np.ndarray,
        l2_normalize_text: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        [BOTH PATHS] Normalize features using fitted scalers.

        Args:
            egemaps: eGeMAPS features (N, 88)
            mfcc: MFCC features (N, 120)
            text_embeddings: Text embeddings (N, 384)
            l2_normalize_text: Whether to L2-normalize text embeddings

        Returns:
            Dict with keys: 'egemaps', 'mfcc', 'text_embeddings' (all normalized)
        """
        if not self._fitted:
            raise RuntimeError(
                "Normalizer not fitted. Call fit() (training) or load() (inference) first."
            )

        # [DATA_FLOW] StandardScale eGeMAPS
        egemaps_norm = (
            (egemaps - self._scalers['egemaps']['mean'])
            / self._scalers['egemaps']['std']
        ).astype(np.float32)

        # [DATA_FLOW] StandardScale MFCC
        mfcc_norm = (
            (mfcc - self._scalers['mfcc']['mean'])
            / self._scalers['mfcc']['std']
        ).astype(np.float32)

        # [DATA_FLOW] L2 normalize text (SBERT outputs are already mostly normalized,
        # but we enforce it for consistency)
        if l2_normalize_text:
            norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # avoid div by zero
            text_norm = (text_embeddings / norms).astype(np.float32)
        else:
            text_norm = text_embeddings.astype(np.float32)

        logger.debug(
            f"[DATA_FLOW] Normalized: egemaps={egemaps_norm.shape}, "
            f"mfcc={mfcc_norm.shape}, text={text_norm.shape}"
        )

        return {
            'egemaps': egemaps_norm,
            'mfcc': mfcc_norm,
            'text_embeddings': text_norm,
        }

    def fit_transform(
        self,
        egemaps: np.ndarray,
        mfcc: np.ndarray,
        text_embeddings: np.ndarray,
        l2_normalize_text: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        [TRAINING_PATH] Fit on data and transform in one step.
        Only use on training split.
        """
        self.fit(egemaps, mfcc)
        return self.transform(egemaps, mfcc, text_embeddings, l2_normalize_text)

    # =========================================================
    # PERSISTENCE: Save/Load scalers for inference parity
    # =========================================================
    def save(self, path: Union[str, Path, None] = None) -> Path:
        """
        Save fitted scalers to disk.

        Args:
            path: File path for scaler persistence (.pkl).
                  Defaults to checkpoints/scalers/feature_scalers.pkl
                  (matching normalization_config.yaml)

        Returns:
            Path to saved file
        """
        if path is None:
            path = DEFAULT_SCALER_PATH
        if not self._fitted:
            raise RuntimeError("Cannot save: normalizer not fitted yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'scalers': self._scalers,
            'version': '1.0',
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"[CHECKPOINT] Scalers saved to: {path}")
        return path

    def load(self, path: Union[str, Path, None] = None) -> "FeatureNormalizer":
        """
        [INFERENCE_PATH] Load previously fitted scalers from disk.

        Args:
            path: Path to saved scaler file (.pkl).
                  Defaults to checkpoints/scalers/feature_scalers.pkl
                  (matching normalization_config.yaml)

        Returns:
            self (for chaining)
        """
        if path is None:
            path = DEFAULT_SCALER_PATH
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        with open(path, 'rb') as f:
            save_data = _safe_pickle_load(f)

        # [VALIDATION_CHECK] Verify structure
        if 'scalers' not in save_data:
            raise ValueError(f"Invalid scaler file: missing 'scalers' key")

        self._scalers = save_data['scalers']

        for name in ['egemaps', 'mfcc']:
            if name not in self._scalers:
                raise ValueError(f"Invalid scaler file: missing '{name}' scaler")
            if 'mean' not in self._scalers[name] or 'std' not in self._scalers[name]:
                raise ValueError(f"Invalid scaler file: '{name}' missing mean/std")

        self._fitted = True
        logger.info(
            f"[INFERENCE_PATH] Scalers loaded from: {path} "
            f"(version={save_data.get('version', 'unknown')})"
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_scaler_stats(self) -> Dict:
        """Return scaler statistics for debugging."""
        if not self._fitted:
            return {}
        return {
            name: {
                'mean_range': (float(s['mean'].min()), float(s['mean'].max())),
                'std_range': (float(s['std'].min()), float(s['std'].max())),
                'dim': len(s['mean']),
            }
            for name, s in self._scalers.items()
        }
