"""
[LAYER_START] Session 5: Depression Dataset
PyTorch Dataset for subject-level sequences of fused feature chunks.

Training path: Load fused features per subject + PHQ-8 labels from split CSVs
Inference path: Accept pre-fused feature sequence for a single subject
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DepressionDataset(Dataset):
    """
    PyTorch Dataset: each item is one subject's sequence of fused chunks + PHQ-8 label.

    Supports chunk-level augmentation for training (disabled for val/test):
        - Temporal dropout: randomly drop chunks to force robustness
        - Feature noise: small Gaussian noise to prevent memorization
        - Label smoothing: small noise on PHQ scores to reduce overfitting

    Returns per __getitem__:
        features: Tensor (num_chunks, D) - variable length per subject
        label: Tensor scalar - PHQ-8 score (float)
        length: int - actual number of chunks (before padding)
        participant_id: str
    """

    def __init__(
        self,
        participant_features: Dict[str, np.ndarray],
        labels: Dict[str, float],
        participant_ids: Optional[List[str]] = None,
        max_chunks: int = 0,
        augment: bool = False,
        temporal_dropout_rate: float = 0.15,
        feature_noise_std: float = 0.01,
        label_noise_std: float = 0.3,
        depression_threshold: float = 10.0,
        item_labels: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Args:
            participant_features: {participant_id: np.ndarray (num_chunks, D)}
            labels: {participant_id: phq8_score}
            participant_ids: Optional subset of IDs to include (filters both dicts)
            max_chunks: Max chunks per subject (0 = no limit). Truncates from end.
            augment: Enable chunk-level augmentation (training only)
            temporal_dropout_rate: Base fraction of chunks to randomly drop (0-0.3)
            feature_noise_std: Base std of Gaussian noise added to features
            label_noise_std: Std of Gaussian noise added to PHQ labels
            depression_threshold: PHQ score threshold for class-aware augmentation
        """
        self.augment = augment
        self.temporal_dropout_rate = temporal_dropout_rate
        self.feature_noise_std = feature_noise_std
        self.label_noise_std = label_noise_std
        self.depression_threshold = depression_threshold
        self.item_labels = item_labels or {}
        # Filter to requested IDs if provided
        if participant_ids is not None:
            ids = [pid for pid in participant_ids if pid in participant_features]
        else:
            ids = list(participant_features.keys())

        # [VALIDATION_CHECK] Verify all IDs have both features and labels
        self.samples = []
        missing_labels = []
        missing_features = []

        for pid in ids:
            if pid not in participant_features:
                missing_features.append(pid)
                continue
            if pid not in labels:
                missing_labels.append(pid)
                continue

            feats = participant_features[pid]
            if max_chunks > 0 and feats.shape[0] > max_chunks:
                feats = feats[:max_chunks]

            self.samples.append({
                'participant_id': str(pid),
                'features': feats.astype(np.float32),
                'label': float(labels[pid]),
                'length': feats.shape[0],
            })

        if missing_labels:
            logger.warning(
                f"[VALIDATION_CHECK] {len(missing_labels)} participants missing labels, skipped"
            )
        if missing_features:
            logger.warning(
                f"[VALIDATION_CHECK] {len(missing_features)} participants missing features, skipped"
            )

        logger.info(
            f"[LAYER_START] DepressionDataset: {len(self.samples)} subjects, "
            f"chunks/subject: min={min(s['length'] for s in self.samples) if self.samples else 0}, "
            f"max={max(s['length'] for s in self.samples) if self.samples else 0}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        features = sample['features'].copy()  # copy to avoid mutating stored data
        label = sample['label']
        length = sample['length']

        if self.augment:
            is_depressed = label >= self.depression_threshold
            # Use global numpy RNG so augmentation varies across epochs.
            # Worker seeds (MED-3) still ensure reproducibility within an epoch.
            features, label = self._augment(features, label, strong=is_depressed, rng=np.random)
            length = features.shape[0]

        result = {
            'features': torch.from_numpy(features),                      # (T, D)
            'label': torch.tensor(label, dtype=torch.float32),           # scalar
            'length': length,                                            # int
            'participant_id': sample['participant_id'],                  # str
        }

        # Multi-task targets (if per-item labels available)
        pid = sample['participant_id']
        if pid in self.item_labels:
            result['item_scores'] = torch.tensor(self.item_labels[pid], dtype=torch.float32)  # (8,)
            result['binary_label'] = torch.tensor(1.0 if label >= self.depression_threshold else 0.0, dtype=torch.float32)
        else:
            result['item_scores'] = torch.zeros(8, dtype=torch.float32)
            result['binary_label'] = torch.tensor(1.0 if label >= self.depression_threshold else 0.0, dtype=torch.float32)

        return result

    def _augment(self, features: np.ndarray, label: float, strong: bool = False, rng: np.random.RandomState = None):
        """Apply class-aware chunk-level augmentations (training only).

        Depressed participants (strong=True) get stronger augmentation to
        create more diverse training views of the minority class:
          - Mild (non-depressed): temporal_dropout=10%, noise_std=0.02
          - Strong (depressed):   temporal_dropout=20%, noise_std=0.05, temporal_shift

        This balances effective diversity:
          ~84 non-depressed × 2× diversity ≈ 168 effective views
          ~23 depressed × 5× diversity ≈ 115 effective views
          → ratio improves from 3.6:1 to 1.5:1

        Args:
            features: (T, D) numpy array
            label: PHQ-8 score
            strong: if True, apply stronger augmentation (minority class)
            rng: numpy RandomState for reproducibility (MED-3)
        """
        if rng is None:
            rng = np.random.RandomState()
        T, D = features.shape

        # 1. Temporal dropout: randomly remove chunks
        # Use config value as base, double for depressed (strong) class
        drop_rate = self.temporal_dropout_rate * 2 if strong else self.temporal_dropout_rate
        if T > 5 and drop_rate > 0:
            keep_mask = rng.rand(T) > drop_rate
            # Always keep at least 40% of chunks
            min_keep = max(int(T * 0.4), 3)
            if keep_mask.sum() < min_keep:
                keep_indices = rng.choice(T, min_keep, replace=False)
                keep_mask[keep_indices] = True
            features = features[keep_mask]

        # 2. Feature noise: stronger for minority class
        # Use config value as base, multiply by 2.5 for depressed class
        noise_std = self.feature_noise_std * 2.5 if strong else self.feature_noise_std
        noise = rng.normal(0, noise_std, features.shape).astype(np.float32)
        features = features + noise

        # 3. Temporal shift (minority only): swap adjacent chunks
        #    Tests if model relies too heavily on exact temporal order
        if strong and features.shape[0] > 4:
            n_swaps = max(1, features.shape[0] // 10)
            for _ in range(n_swaps):
                i = rng.randint(0, features.shape[0] - 1)
                features[[i, i + 1]] = features[[i + 1, i]]

        # 4. Label noise: smooth PHQ scores slightly
        if self.label_noise_std > 0:
            label = label + rng.normal(0, self.label_noise_std)
            label = float(np.clip(label, 0, 24))  # PHQ-8 range

        # 5. Feature masking (SpecAugment-style): zero out contiguous feature dims
        #    Use small mask to avoid corrupting statistics pooling
        T_cur, D = features.shape
        mask_frac = 0.08 if strong else 0.04
        mask_width = max(1, int(D * mask_frac))
        start = rng.randint(0, max(D - mask_width, 1))
        features[:, start:start + mask_width] = 0.0

        return features, label

    @classmethod
    def from_split_csv(
        cls,
        split_csv: Union[str, Path],
        participant_features: Dict[str, np.ndarray],
        label_column: str = "PHQ_Score",
        id_column: str = "Participant_ID",
        max_chunks: int = 0,
        augment: bool = False,
        temporal_dropout_rate: float = 0.15,
        feature_noise_std: float = 0.01,
        label_noise_std: float = 0.3,
        depression_threshold: float = 10.0,
        detailed_labels_csv: Optional[Union[str, Path]] = None,
    ) -> "DepressionDataset":
        """
        [TRAINING_PATH] Build dataset from a split CSV (train/dev/test).

        Args:
            split_csv: Path to split CSV with participant IDs and PHQ scores
            participant_features: {participant_id: fused_features (num_chunks, D)}
            label_column: Column name for PHQ-8 score
            id_column: Column name for participant ID
            max_chunks: Max chunks per subject
            augment: Enable augmentation (training only)
            temporal_dropout_rate: Fraction of chunks to drop randomly
            feature_noise_std: Std of feature noise
            label_noise_std: Std of label noise
            depression_threshold: PHQ score threshold for class-aware augmentation

        Returns:
            DepressionDataset instance
        """
        split_csv = Path(split_csv)
        df = pd.read_csv(split_csv)

        # Build labels dict
        labels = {}
        for _, row in df.iterrows():
            pid = str(int(row[id_column]))
            labels[pid] = float(row[label_column])

        participant_ids = list(labels.keys())

        # Load per-item PHQ-8 labels if available
        item_labels = {}
        if detailed_labels_csv is not None:
            detailed_path = Path(detailed_labels_csv)
            if detailed_path.exists():
                detail_df = pd.read_csv(detailed_path)
                item_cols = [c for c in detail_df.columns if c.startswith("PHQ_8") and c != "PHQ_8Total"]
                for _, row in detail_df.iterrows():
                    pid = str(int(row["Participant_ID"]))
                    if pid in labels:
                        item_labels[pid] = [float(row[c]) for c in item_cols]
                logger.info(f"[TRAINING_PATH] Loaded per-item labels for {len(item_labels)} participants")

        logger.info(
            f"[TRAINING_PATH] Loading split: {split_csv.name}, "
            f"{len(participant_ids)} participants, "
            f"PHQ range: [{df[label_column].min()}, {df[label_column].max()}]"
        )

        return cls(
            participant_features=participant_features,
            labels=labels,
            participant_ids=participant_ids,
            max_chunks=max_chunks,
            augment=augment,
            temporal_dropout_rate=temporal_dropout_rate,
            feature_noise_std=feature_noise_std,
            label_noise_std=label_noise_std,
            depression_threshold=depression_threshold,
            item_labels=item_labels,
        )

    @classmethod
    def for_inference(
        cls,
        participant_id: str,
        features: np.ndarray,
        label: float = -1.0,
    ) -> "DepressionDataset":
        """
        [INFERENCE_PATH] Build dataset for a single subject.

        Args:
            participant_id: Subject identifier
            features: Fused features (num_chunks, 592)
            label: PHQ-8 score if known (-1.0 = unknown)

        Returns:
            DepressionDataset with single subject
        """
        return cls(
            participant_features={participant_id: features},
            labels={participant_id: label},
        )
