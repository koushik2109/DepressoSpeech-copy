"""
[LAYER_START] Session 6 (Revised): PCA Dimensionality Reduction
Reduces fused 592-dim features to 64-dim before model input.

Training path: Fit PCA on train split → transform train/dev/test → save
Inference path: Load saved PCA → transform new features
"""

import io
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Union, Optional

import sklearn

# SEC-1b: Restrict unpickling to known safe types only
_SAFE_MODULES = frozenset({
    'numpy', 'numpy.core.multiarray', 'numpy.core.numeric',
    'numpy.core._multiarray_umath', 'collections', 'builtins',
    'sklearn', 'sklearn.decomposition', 'sklearn.decomposition._pca',
    'sklearn.utils._bunch', 'scipy', 'scipy.sparse',
})

class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.split('.')[0] in _SAFE_MODULES or module in _SAFE_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked unpickling of {module}.{name} — not in allowlist"
        )

def _safe_pickle_load(f: io.BufferedIOBase):
    return _RestrictedUnpickler(f).load()
from sklearn.decomposition import PCA

from src.features.constants import FUSED_DIM

logger = logging.getLogger(__name__)

EXPECTED_INPUT_DIM = FUSED_DIM
DEFAULT_PCA_DIR = "checkpoints/scalers"
DEFAULT_PCA_FILENAME = "pca_reducer.pkl"
DEFAULT_PCA_PATH = str(Path(DEFAULT_PCA_DIR) / DEFAULT_PCA_FILENAME)


class PCAReducer:
    """
    PCA for reducing fused feature dimensionality (592 → n_components).

    Fitted on training split only, applied to all splits + inference.
    Saves/loads for inference parity (same as FeatureNormalizer pattern).

    Uses sklearn.decomposition.PCA internally for numerical stability
    and consistent API.
    """

    def __init__(self, n_components: int = 64, expected_dim: int = EXPECTED_INPUT_DIM):
        self.n_components = n_components
        self.expected_dim = expected_dim
        self._pca: Optional[PCA] = None
        self._fitted = False

    def _validate_input(self, X: np.ndarray, context: str = "") -> None:
        """Validate input shape and dimensionality."""
        if X.ndim != 2:
            raise ValueError(
                f"[VALIDATION_CHECK] {context} Expected 2D array, got {X.ndim}D"
            )
        if X.shape[1] != self.expected_dim:
            raise ValueError(
                f"[VALIDATION_CHECK] {context} Expected {self.expected_dim} features, "
                f"got {X.shape[1]}. Check feature pipeline output."
            )

    # =========================================================
    # TRAINING PATH: Fit PCA on training data
    # =========================================================
    def fit(self, X: np.ndarray) -> "PCAReducer":
        """
        [TRAINING_PATH] Fit PCA on training split fused features.

        Args:
            X: Training fused features (N_train, 592)

        Returns:
            self (for chaining)
        """
        self._validate_input(X, context="fit()")
        N, D = X.shape

        effective_components = min(self.n_components, N, D)
        if effective_components < self.n_components:
            logger.warning(
                f"[VALIDATION_CHECK] Capping n_components from {self.n_components} "
                f"to {effective_components} (N={N}, D={D})"
            )

        self._pca = PCA(n_components=effective_components)
        self._pca.fit(X.astype(np.float64))
        self.n_components = effective_components
        self._fitted = True

        # MED-8: Compute training distribution stats for OOD detection
        X_reduced = self._pca.transform(X.astype(np.float64)).astype(np.float32)
        self._train_mean = X_reduced.mean(axis=0)
        self._train_std = X_reduced.std(axis=0)
        self._train_std[self._train_std == 0] = 1.0  # avoid div by zero

        cumulative = self._pca.explained_variance_ratio_.cumsum()
        logger.info(
            f"[TRAINING_PATH] PCA fitted: {D} → {self.n_components} components, "
            f"explained variance: {cumulative[-1]*100:.1f}% "
            f"(top-10: {cumulative[min(9, len(cumulative)-1)]*100:.1f}%)"
        )
        return self

    # =========================================================
    # BOTH PATHS: Transform features
    # =========================================================
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        [BOTH PATHS] Project features to PCA space.

        Args:
            X: Fused features (N, 592)

        Returns:
            np.ndarray (N, n_components) — reduced features (float32)
        """
        if not self._fitted:
            raise RuntimeError(
                "PCA not fitted. Call fit() (training) or load() (inference) first."
            )
        self._validate_input(X, context="transform()")

        X_reduced = self._pca.transform(X.astype(np.float64)).astype(np.float32)

        logger.debug(
            f"[DATA_FLOW] PCA transform: {X.shape} → {X_reduced.shape}"
        )
        return X_reduced

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """[TRAINING_PATH] Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    # =========================================================
    # PERSISTENCE: Save/Load for inference parity
    # =========================================================
    def save(self, path: Union[str, Path, None] = None) -> Path:
        """Save fitted PCA to disk."""
        if not self._fitted:
            raise RuntimeError("Cannot save: PCA not fitted yet.")

        if path is None:
            path = DEFAULT_PCA_PATH
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'pca': self._pca,
            'n_components': self.n_components,
            'expected_dim': self.expected_dim,
            'version': '2.1',
            'sklearn_version': sklearn.__version__,
            'train_mean': getattr(self, '_train_mean', None),
            'train_std': getattr(self, '_train_std', None),
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"[CHECKPOINT] PCA saved to: {path}")
        return path

    def load(self, path: Union[str, Path, None] = None) -> "PCAReducer":
        """[INFERENCE_PATH] Load previously fitted PCA from disk."""
        if path is None:
            path = DEFAULT_PCA_PATH
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PCA file not found: {path}")

        with open(path, 'rb') as f:
            data = _safe_pickle_load(f)

        self._pca = data['pca']
        self.n_components = data['n_components']
        self.expected_dim = data.get('expected_dim', EXPECTED_INPUT_DIM)
        self._train_mean = data.get('train_mean')
        self._train_std = data.get('train_std')
        self._fitted = True

        # Warn if sklearn version differs (PCA pickle may be incompatible)
        saved_sklearn = data.get('sklearn_version')
        if saved_sklearn and saved_sklearn != sklearn.__version__:
            logger.warning(
                f"[VALIDATION_CHECK] PCA was saved with sklearn {saved_sklearn}, "
                f"current is {sklearn.__version__}. If errors occur, retrain PCA."
            )

        logger.info(
            f"[INFERENCE_PATH] PCA loaded from: {path}, "
            f"n_components={self.n_components}"
        )
        return self

    @property
    def explained_variance_total(self) -> float:
        """Total explained variance ratio (0-1)."""
        if not self._fitted or self._pca is None:
            return 0.0
        return float(self._pca.explained_variance_ratio_.sum())

    def compute_ood_score(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Compute per-sample OOD score as mean absolute z-score (MED-8).

        Samples with high z-scores are far from the training distribution
        and predictions should be treated with lower confidence.

        Args:
            X_reduced: (N, n_components) PCA-reduced features

        Returns:
            (N,) array of OOD scores (higher = more out-of-distribution)
        """
        if self._train_mean is None or self._train_std is None:
            logger.warning("[MED-8] OOD detection unavailable: no training stats saved")
            return np.zeros(X_reduced.shape[0])
        z_scores = np.abs((X_reduced - self._train_mean) / self._train_std)
        return z_scores.mean(axis=1)
