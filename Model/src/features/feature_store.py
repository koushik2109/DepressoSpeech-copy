"""
[LAYER_START] Feature Extraction - Feature Store
Handles saving and loading extracted features in .npz format.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature store for saving/loading extracted features.

    Storage format: .npz (numpy compressed archive)
    Naming: {participant_id}_{source}.npz
    Source: 'training' or 'inference'
    """

    def __init__(self, store_dir: str = "data/features"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[LAYER_START] FeatureStore initialized at: {self.store_dir}")

    def _get_path(self, participant_id: str, source: str = "training") -> Path:
        """Get file path for a participant's features."""
        return self.store_dir / f"{participant_id}_{source}.npz"

    def save(
        self,
        participant_id: str,
        egemaps: np.ndarray,
        mfcc: np.ndarray,
        text_embeddings: np.ndarray,
        source: str = "training",
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save features for a participant.

        Args:
            participant_id: Unique participant identifier
            egemaps: eGeMAPS features (N, 88)
            mfcc: MFCC features (N, 120)
            text_embeddings: Text embeddings (N, 384)
            source: 'training' or 'inference'
            metadata: Optional metadata dict

        Returns:
            Path to saved file
        """
        filepath = self._get_path(participant_id, source)

        save_dict = {
            'egemaps': egemaps,
            'mfcc': mfcc,
            'text_embeddings': text_embeddings,
            'source': np.array([source]),
        }
        if metadata:
            for key, value in metadata.items():
                save_dict[f'meta_{key}'] = np.array([value])

        np.savez_compressed(filepath, **save_dict)
        logger.info(
            f"[DATA_FLOW] Saved features for {participant_id} ({source}): "
            f"egemaps={egemaps.shape}, mfcc={mfcc.shape}, text={text_embeddings.shape}"
        )
        return filepath

    def load(
        self, participant_id: str, source: str = "training"
    ) -> Dict[str, np.ndarray]:
        """
        Load features for a participant.

        Returns:
            Dict with keys: 'egemaps', 'mfcc', 'text_embeddings', 'source'
        """
        filepath = self._get_path(participant_id, source)
        if not filepath.exists():
            raise FileNotFoundError(
                f"No features found for {participant_id} ({source}) at {filepath}"
            )

        data = np.load(filepath, allow_pickle=False)
        result = {
            'egemaps': data['egemaps'],
            'mfcc': data['mfcc'],
            'text_embeddings': data['text_embeddings'],
            'source': str(data['source'][0]) if 'source' in data else source,
        }

        # Load metadata
        for key in data.files:
            if key.startswith('meta_'):
                result[key] = data[key][0]

        logger.info(
            f"[DATA_FLOW] Loaded features for {participant_id} ({source}): "
            f"egemaps={result['egemaps'].shape}, mfcc={result['mfcc'].shape}, "
            f"text={result['text_embeddings'].shape}"
        )
        return result

    def exists(self, participant_id: str, source: str = "training") -> bool:
        """Check if features exist for a participant."""
        return self._get_path(participant_id, source).exists()

    def list_participants(self, source: Optional[str] = None) -> list:
        """List all participant IDs with saved features."""
        participants = []
        for f in self.store_dir.glob("*.npz"):
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                pid, src = parts
                if source is None or src == source:
                    participants.append(pid)
        return sorted(set(participants))
