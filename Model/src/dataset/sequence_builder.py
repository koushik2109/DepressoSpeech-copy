"""
[LAYER_START] Session 5: Sequence Builder
Orchestrates building DataLoaders from split CSVs + fused features.

Training path: Build train/dev/test DataLoaders from split CSVs
Inference path: Build single-subject DataLoader from fused features
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset.depression_dataset import DepressionDataset
from src.dataset.collate import depression_collate_fn


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducible augmentation (MED-3)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    Builds DataLoaders for training and inference from fused feature arrays.

    Training: split CSVs + participant_features dict → DataLoaders
    Inference: single subject features → DataLoader (batch_size=1)
    """

    def __init__(
        self,
        batch_size: int = 8,
        max_chunks: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        phq_threshold: float = 10.0,
        temporal_dropout_rate: float = 0.05,
        feature_noise_std: float = 0.005,
        label_noise_std: float = 0.15,
        depression_threshold: float = 10.0,
        detailed_labels_csv: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.max_chunks = max_chunks
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.phq_threshold = phq_threshold
        self.temporal_dropout_rate = temporal_dropout_rate
        self.feature_noise_std = feature_noise_std
        self.label_noise_std = label_noise_std
        self.depression_threshold = depression_threshold
        self.detailed_labels_csv = detailed_labels_csv

    # =========================================================
    # TRAINING PATH: Build train/dev/test DataLoaders
    # =========================================================
    def build_train_loaders(
        self,
        participant_features: Dict[str, np.ndarray],
        train_csv: Union[str, Path],
        dev_csv: Union[str, Path],
        test_csv: Optional[Union[str, Path]] = None,
        label_column: str = "PHQ_Score",
        id_column: str = "Participant_ID",
    ) -> Dict[str, DataLoader]:
        """
        [TRAINING_PATH] Build DataLoaders for all splits.

        Args:
            participant_features: {participant_id: fused_features (T, 592)}
            train_csv: Path to training split CSV
            dev_csv: Path to dev split CSV
            test_csv: Optional path to test split CSV
            label_column: PHQ score column name
            id_column: Participant ID column name

        Returns:
            Dict with keys 'train', 'dev', optionally 'test' → DataLoader
        """
        loaders = {}

        # Train loader with WeightedRandomSampler for class balance
        train_dataset = DepressionDataset.from_split_csv(
            split_csv=train_csv,
            participant_features=participant_features,
            label_column=label_column,
            id_column=id_column,
            max_chunks=self.max_chunks,
            augment=True,  # chunk-level augmentation for training only
            temporal_dropout_rate=self.temporal_dropout_rate,
            feature_noise_std=self.feature_noise_std,
            label_noise_std=self.label_noise_std,
            depression_threshold=self.depression_threshold,
            detailed_labels_csv=self.detailed_labels_csv,
        )

        # Build sampler: inverse-frequency weighting by depressed/not-depressed
        train_sampler = self._build_weighted_sampler(
            train_dataset, train_csv, label_column, id_column
        )

        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,  # replaces shuffle=True
            collate_fn=depression_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )

        # Dev loader (shuffle=False)
        dev_dataset = DepressionDataset.from_split_csv(
            split_csv=dev_csv,
            participant_features=participant_features,
            label_column=label_column,
            id_column=id_column,
            max_chunks=self.max_chunks,
            detailed_labels_csv=self.detailed_labels_csv,
        )
        loaders['dev'] = DataLoader(
            dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=depression_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        # Test loader (optional)
        if test_csv is not None:
            test_dataset = DepressionDataset.from_split_csv(
                split_csv=test_csv,
                participant_features=participant_features,
                label_column=label_column,
                id_column=id_column,
                max_chunks=self.max_chunks,
            )
            loaders['test'] = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=depression_collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        logger.info(
            f"[TRAINING_PATH] DataLoaders built: "
            + ", ".join(f"{k}={len(v.dataset)} subjects" for k, v in loaders.items())
        )
        return loaders

    # =========================================================
    # TRAINING PATH: Weighted sampler for class balance
    # =========================================================
    def _build_weighted_sampler(
        self,
        dataset: DepressionDataset,
        split_csv: Union[str, Path],
        label_column: str,
        id_column: str,
    ) -> WeightedRandomSampler:
        """
        Build WeightedRandomSampler using inverse-frequency class weights.

        Depressed (PHQ >= threshold) and not-depressed (PHQ < threshold)
        are sampled with equal probability per epoch, counteracting the
        60-65% non-depressed / 35-40% depressed imbalance.

        Args:
            dataset: Training DepressionDataset
            split_csv: Path to split CSV for reading labels
            label_column: PHQ score column name
            id_column: Participant ID column name

        Returns:
            WeightedRandomSampler instance
        """
        df = pd.read_csv(split_csv)
        label_map = {
            str(int(row[id_column])): float(row[label_column])
            for _, row in df.iterrows()
        }

        # Compute per-sample weights
        sample_weights = []
        n_depressed = 0
        n_not_depressed = 0

        for sample in dataset.samples:
            pid = sample['participant_id']
            score = label_map.get(pid, sample['label'])
            if score >= self.phq_threshold:
                n_depressed += 1
            else:
                n_not_depressed += 1

        # Inverse frequency: rarer class gets higher weight
        total = n_depressed + n_not_depressed
        w_depressed = total / (2.0 * max(n_depressed, 1))
        w_not_depressed = total / (2.0 * max(n_not_depressed, 1))

        for sample in dataset.samples:
            pid = sample['participant_id']
            score = label_map.get(pid, sample['label'])
            if score >= self.phq_threshold:
                sample_weights.append(w_depressed)
            else:
                sample_weights.append(w_not_depressed)

        weights_tensor = torch.FloatTensor(sample_weights)

        logger.info(
            f"[TRAINING_PATH] WeightedRandomSampler: "
            f"depressed={n_depressed} (w={w_depressed:.2f}), "
            f"not_depressed={n_not_depressed} (w={w_not_depressed:.2f}), "
            f"threshold={self.phq_threshold}"
        )

        return WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(weights_tensor),
            replacement=True,
        )

    # =========================================================
    # INFERENCE PATH: Single subject DataLoader
    # =========================================================
    def build_inference_loader(
        self,
        participant_id: str,
        features: np.ndarray,
        label: float = -1.0,
    ) -> DataLoader:
        """
        [INFERENCE_PATH] Build DataLoader for a single subject.

        Args:
            participant_id: Subject identifier
            features: Fused features (num_chunks, 592)
            label: PHQ-8 score if known (-1.0 = unknown)

        Returns:
            DataLoader with batch_size=1
        """
        dataset = DepressionDataset.for_inference(
            participant_id=participant_id,
            features=features,
            label=label,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=depression_collate_fn,
            num_workers=0,
            pin_memory=False,
        )
        logger.info(
            f"[INFERENCE_PATH] Inference loader: participant={participant_id}, "
            f"chunks={features.shape[0]}, dim={features.shape[1]}"
        )
        return loader
