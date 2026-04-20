"""
Multimodal Dataset for Gated Fusion Model (V2).

Each sample has separate text and audio feature tensors + quality scores.
Unlike V1 which concatenated everything into a single fused vector,
V2 keeps modalities separate for the gated fusion model.
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for gated multimodal fusion.

    Each item returns:
        text_features:  (T, 384) SBERT embeddings
        audio_features: (T, 768) HuBERT embeddings
        audio_quality:  (T,) quality scores [0, 1]
        label:          scalar PHQ-8 score
        length:         int number of chunks
        participant_id: str
    """

    def __init__(
        self,
        participant_data: Dict[str, Dict[str, np.ndarray]],
        labels: Dict[str, float],
        participant_ids: Optional[List[str]] = None,
        augment: bool = False,
        temporal_dropout_rate: float = 0.1,
        feature_noise_std: float = 0.005,
        label_noise_std: float = 0.15,
        depression_threshold: float = 10.0,
    ):
        """
        Args:
            participant_data: {pid: {'text': (T,384), 'audio': (T,768), 'quality': (T,)}}
            labels: {pid: phq8_score}
            participant_ids: Optional subset filter
            augment: Enable training augmentations
        """
        self.augment = augment
        self.temporal_dropout_rate = temporal_dropout_rate
        self.feature_noise_std = feature_noise_std
        self.label_noise_std = label_noise_std
        self.depression_threshold = depression_threshold

        if participant_ids is not None:
            ids = [pid for pid in participant_ids if pid in participant_data]
        else:
            ids = list(participant_data.keys())

        self.samples = []
        for pid in ids:
            if pid not in labels:
                continue
            data = participant_data[pid]
            self.samples.append({
                'participant_id': str(pid),
                'text': data['text'].astype(np.float32),
                'audio': data['audio'].astype(np.float32),
                'quality': data['quality'].astype(np.float32),
                'label': float(labels[pid]),
                'length': data['text'].shape[0],
            })

        logger.info(
            f"[LAYER_START] MultimodalDataset: {len(self.samples)} subjects, "
            f"chunks/subject: min={min(s['length'] for s in self.samples) if self.samples else 0}, "
            f"max={max(s['length'] for s in self.samples) if self.samples else 0}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        text = sample['text'].copy()
        audio = sample['audio'].copy()
        quality = sample['quality'].copy()
        label = sample['label']
        length = sample['length']

        if self.augment:
            text, audio, quality, label = self._augment(
                text, audio, quality, label,
                strong=label >= self.depression_threshold,
            )
            length = text.shape[0]

        return {
            'text_features': torch.from_numpy(text),         # (T, 384)
            'audio_features': torch.from_numpy(audio),       # (T, 768)
            'audio_quality': torch.from_numpy(quality),      # (T,)
            'label': torch.tensor(label, dtype=torch.float32),
            'length': length,
            'participant_id': sample['participant_id'],
        }

    def _augment(self, text, audio, quality, label, strong=False):
        """Apply augmentations consistently across modalities."""
        T = text.shape[0]
        rng = np.random

        # 1. Temporal dropout (same indices for both modalities)
        drop_rate = self.temporal_dropout_rate * 2 if strong else self.temporal_dropout_rate
        if T > 5 and drop_rate > 0:
            keep_mask = rng.rand(T) > drop_rate
            min_keep = max(int(T * 0.4), 3)
            if keep_mask.sum() < min_keep:
                keep_idx = rng.choice(T, min_keep, replace=False)
                keep_mask[keep_idx] = True
            text = text[keep_mask]
            audio = audio[keep_mask]
            quality = quality[keep_mask]

        # 2. Feature noise
        noise_std = self.feature_noise_std * 2 if strong else self.feature_noise_std
        text = text + rng.normal(0, noise_std, text.shape).astype(np.float32)
        audio = audio + rng.normal(0, noise_std, audio.shape).astype(np.float32)

        # 3. Label noise
        if self.label_noise_std > 0:
            label = float(np.clip(label + rng.normal(0, self.label_noise_std), 0, 24))

        return text, audio, quality, label

    @classmethod
    def from_split_csv(
        cls,
        split_csv: Union[str, Path],
        participant_data: Dict[str, Dict[str, np.ndarray]],
        label_column: str = "PHQ_Score",
        id_column: str = "Participant_ID",
        **kwargs,
    ) -> "MultimodalDataset":
        """Build dataset from split CSV + pre-loaded features."""
        df = pd.read_csv(split_csv)
        labels = {
            str(int(row[id_column])): float(row[label_column])
            for _, row in df.iterrows()
        }
        pids = [str(int(row[id_column])) for _, row in df.iterrows()]
        return cls(
            participant_data=participant_data,
            labels=labels,
            participant_ids=pids,
            **kwargs,
        )


def multimodal_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate multimodal samples with padding.

    Pads text, audio, quality to max sequence length in batch.
    Returns mask for valid positions.
    """
    # Sort by length descending (for pack_padded_sequence compatibility)
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    max_len = batch[0]['length']
    B = len(batch)

    text_dim = batch[0]['text_features'].shape[1]
    audio_dim = batch[0]['audio_features'].shape[1]

    text_padded = torch.zeros(B, max_len, text_dim)
    audio_padded = torch.zeros(B, max_len, audio_dim)
    quality_padded = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, sample in enumerate(batch):
        L = sample['length']
        text_padded[i, :L] = sample['text_features'][:L]
        audio_padded[i, :L] = sample['audio_features'][:L]
        quality_padded[i, :L] = sample['audio_quality'][:L]
        mask[i, :L] = True
        labels[i] = sample['label']
        lengths[i] = L

    return {
        'text_features': text_padded,
        'audio_features': audio_padded,
        'audio_quality': quality_padded,
        'mask': mask,
        'labels': labels,
        'lengths': lengths,
        'participant_ids': [s['participant_id'] for s in batch],
    }
