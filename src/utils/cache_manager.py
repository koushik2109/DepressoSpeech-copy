"""Cache manager for acoustic feature persistence.

Manages a JSON-based cache index that maps audio filenames to their
precomputed feature files, avoiding redundant feature extraction.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_cache_index(index_path: str) -> Dict[str, str]:
    """Load the feature cache index from disk.

    Args:
        index_path: Path to the feature_index.json file.

    Returns:
        Dictionary mapping audio filenames to cached feature filenames.
    """
    path = Path(index_path)
    if not path.exists():
        logger.info("Cache index not found at %s, returning empty index.", index_path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        index: Dict[str, str] = json.load(f)

    logger.info("Loaded cache index with %d entries.", len(index))
    return index


def save_cache_index(index: Dict[str, str], index_path: str) -> None:
    """Persist the cache index to disk.

    Args:
        index: Dictionary mapping audio filenames to cached feature filenames.
        index_path: Path to the feature_index.json file.
    """
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    logger.info("Saved cache index with %d entries to %s.", len(index), index_path)


def check_cache(
    audio_filename: str,
    index: Dict[str, str],
    cache_dir: str,
) -> bool:
    """Check whether cached features exist for a given audio file.

    Args:
        audio_filename: Name of the source audio file (e.g. "300_AUDIO.wav").
        index: Current cache index dictionary.
        cache_dir: Directory where cached feature files are stored.

    Returns:
        True if valid cached features exist, False otherwise.
    """
    if audio_filename not in index:
        return False

    feature_file = Path(cache_dir) / index[audio_filename]
    exists = feature_file.exists()
    if not exists:
        logger.warning(
            "Index references %s but file not found on disk.", feature_file
        )
    return exists


def load_cached_features(
    audio_filename: str,
    index: Dict[str, str],
    cache_dir: str,
) -> Optional[Dict[str, Any]]:
    """Load precomputed features from the cache.

    Args:
        audio_filename: Name of the source audio file.
        index: Current cache index dictionary.
        cache_dir: Directory where cached feature files are stored.

    Returns:
        Dictionary of feature arrays, or None if cache miss.
    """
    if not check_cache(audio_filename, index, cache_dir):
        logger.debug("Cache miss for %s.", audio_filename)
        return None

    feature_path = Path(cache_dir) / index[audio_filename]
    with open(feature_path, "rb") as f:
        features: Dict[str, Any] = pickle.load(f)

    logger.info("Loaded cached features for %s.", audio_filename)
    return features


def store_features(
    audio_filename: str,
    participant_id: str,
    features: Dict[str, Any],
    index: Dict[str, str],
    cache_dir: str,
    index_path: str,
) -> str:
    """Compute-once store: save features and update the cache index.

    Args:
        audio_filename: Name of the source audio file (e.g. "300_AUDIO.wav").
        participant_id: Participant identifier (e.g. "300").
        features: Dictionary of extracted feature arrays.
        index: Current cache index dictionary (mutated in place).
        cache_dir: Directory where cached feature files are stored.
        index_path: Path to the feature_index.json file.

    Returns:
        Filename of the stored feature file.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    feature_filename = f"{participant_id}_features.pkl"
    feature_path = cache_path / feature_filename

    with open(feature_path, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Update index
    index[audio_filename] = feature_filename
    save_cache_index(index, index_path)

    logger.info("Stored features for %s → %s.", audio_filename, feature_filename)
    return feature_filename
