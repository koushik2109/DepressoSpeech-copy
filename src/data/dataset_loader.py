"""Dataset loader for the DAIC-WOZ depression interview corpus.

Orchestrates the end-to-end pipeline:
    Audio → Preprocessing → Feature Extraction → Caching → Training Dataset
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from numpy.typing import NDArray

from src.data.preprocessing import preprocess_audio
from src.features.acoustic_features import extract_all_features
from src.utils.cache_manager import (
    check_cache,
    load_cache_index,
    load_cached_features,
    save_cache_index,
    store_features,
)

logger = logging.getLogger(__name__)

# Regex to pull participant ID from typical DAIC-WOZ filenames like "300_AUDIO.wav"
_PARTICIPANT_RE = re.compile(r"^(\d+)_")


def _extract_participant_id(filename: str) -> str:
    """Extract the numeric participant ID from an audio filename.

    Args:
        filename: Audio filename (e.g. "300_AUDIO.wav").

    Returns:
        Participant ID string (e.g. "300").

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    match = _PARTICIPANT_RE.match(filename)
    if match is None:
        raise ValueError(
            f"Cannot extract participant ID from filename: {filename}"
        )
    return match.group(1)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    logger.info("Loaded config from %s.", config_path)
    return config


def discover_audio_files(
    dataset_path: str,
    extension: str = ".wav",
) -> List[Path]:
    """Discover all audio files in the dataset directory.

    Args:
        dataset_path: Root directory of the DAIC-WOZ dataset.
        extension: Audio file extension to search for.

    Returns:
        Sorted list of audio file paths.
    """
    root = Path(dataset_path)
    if not root.exists():
        logger.warning("Dataset path does not exist: %s", dataset_path)
        return []

    files = sorted(root.rglob(f"*{extension}"))
    logger.info("Discovered %d audio files in %s.", len(files), dataset_path)
    return files


def process_single_audio(
    audio_path: Path,
    config: Dict[str, Any],
    cache_index: Dict[str, str],
) -> Dict[str, Any]:
    """Process a single audio file through the full pipeline.

    1. Check the feature cache.
    2. On cache hit → load and return cached features.
    3. On cache miss → preprocess, extract, cache, and return features.

    Args:
        audio_path: Path to the audio file.
        config: Pipeline configuration dictionary.
        cache_index: Current cache index (mutated on cache miss).

    Returns:
        Dictionary of extracted acoustic features.
    """
    filename = audio_path.name
    cache_dir = config["cache_dir"]
    index_path = config["cache_index"]
    target_sr = config.get("sample_rate", 16000)
    top_db = config.get("top_db", 20)
    n_mfcc = config.get("n_mfcc", 13)
    n_fft = config.get("n_fft", 2048)
    hop_length = config.get("hop_length", 512)

    # --- Cache hit path ---
    if check_cache(filename, cache_index, cache_dir):
        logger.info("Cache hit for %s.", filename)
        features = load_cached_features(filename, cache_index, cache_dir)
        if features is not None:
            return features

    # --- Cache miss path ---
    logger.info("Cache miss for %s, running extraction pipeline.", filename)

    # Preprocess
    audio, sr = preprocess_audio(
        str(audio_path), target_sr=target_sr, top_db=top_db
    )

    # Extract features
    features = extract_all_features(
        audio, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    # Store in cache
    participant_id = _extract_participant_id(filename)
    store_features(
        audio_filename=filename,
        participant_id=participant_id,
        features=features,
        index=cache_index,
        cache_dir=cache_dir,
        index_path=index_path,
    )

    return features


def load_dataset(
    config_path: str = "configs/config.yaml",
) -> List[Dict[str, Any]]:
    """Load the full DAIC-WOZ dataset, leveraging feature caching.

    This is the main entry point. It:
        1. Reads the pipeline configuration.
        2. Discovers audio files.
        3. Processes each file (with cache acceleration).
        4. Returns a list of feature dictionaries ready for training.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        List of feature dictionaries, one per audio file.
    """
    config = load_config(config_path)
    dataset_path = config["dataset_path"]
    index_path = config["cache_index"]

    audio_files = discover_audio_files(dataset_path)
    if not audio_files:
        logger.warning("No audio files found. Returning empty dataset.")
        return []

    cache_index = load_cache_index(index_path)
    dataset: List[Dict[str, Any]] = []

    for audio_path in audio_files:
        try:
            features = process_single_audio(audio_path, config, cache_index)
            features["participant_id"] = _extract_participant_id(audio_path.name)
            features["source_file"] = str(audio_path)
            dataset.append(features)
        except Exception:
            logger.exception("Failed to process %s, skipping.", audio_path)

    logger.info(
        "Dataset loading complete: %d / %d files processed.",
        len(dataset),
        len(audio_files),
    )
    return dataset
