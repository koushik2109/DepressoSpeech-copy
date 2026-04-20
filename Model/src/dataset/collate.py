"""
[LAYER_START] Session 5: Collate Function
Pads variable-length sequences and generates attention masks for batching.

[BOTH PATHS] Same collation logic for training and inference.
"""

import torch
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def depression_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    [BOTH PATHS] Collate function for DataLoader.

    Pads variable-length feature sequences to the max length in the batch
    and generates boolean attention masks.

    Args:
        batch: List of dicts from DepressionDataset.__getitem__()
            Each dict has: 'features' (T_i, 592), 'label' (scalar),
                          'length' (int), 'participant_id' (str)

    Returns:
        Dict with:
            'features': Tensor (B, T_max, 592) - zero-padded
            'labels': Tensor (B,) - PHQ-8 scores
            'lengths': Tensor (B,) - actual sequence lengths (int64)
            'mask': Tensor (B, T_max) - True for real chunks, False for padding
            'participant_ids': List[str]
    """
    # Sort by length descending (useful for pack_padded_sequence if needed)
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    lengths = [item['length'] for item in batch]
    max_len = max(lengths)
    feature_dim = batch[0]['features'].shape[1]  # 592
    batch_size = len(batch)

    # [DATA_FLOW] Pad sequences to max_len
    padded_features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    item_scores = torch.zeros(batch_size, 8, dtype=torch.float32)
    binary_labels = torch.zeros(batch_size, dtype=torch.float32)

    participant_ids = []

    for i, item in enumerate(batch):
        seq_len = item['length']
        padded_features[i, :seq_len, :] = item['features'][:seq_len]
        labels[i] = item['label']
        mask[i, :seq_len] = True
        participant_ids.append(item['participant_id'])
        if 'item_scores' in item:
            item_scores[i] = item['item_scores']
        if 'binary_label' in item:
            binary_labels[i] = item['binary_label']

    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)

    logger.debug(
        f"[DATA_FLOW] Collated batch: B={batch_size}, T_max={max_len}, "
        f"D={feature_dim}, lengths={lengths}"
    )

    return {
        'features': padded_features,       # (B, T_max, D)
        'labels': labels,                   # (B,)
        'lengths': lengths_tensor,          # (B,)
        'mask': mask,                       # (B, T_max)
        'participant_ids': participant_ids,  # List[str]
        'item_labels': item_scores,         # (B, 8)
        'binary_labels': binary_labels,     # (B,)
    }
