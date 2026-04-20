#!/usr/bin/env python3
"""
[LAYER_START] Session 7: Training Entry Point
End-to-end: load data → build features → PCA → build dataloaders → train → checkpoint.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training_config.yaml
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.normalizer import FeatureNormalizer
from src.features.fusion import FeatureFusion
from src.features.pca_reducer import PCAReducer
from src.dataset import SequenceBuilder
from src.models import DepressionModel
from src.training import Trainer
from src.utils import setup_logging, RunManager, ExperimentTracker


def load_config(config_path: str) -> dict:
    """Load training config YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_participant_features(
    split_csv: Path,
    feature_dir: Path,
    id_column: str = "Participant_ID",
) -> dict:
    """
    Load pre-extracted per-subject feature arrays from disk.

    Expects: feature_dir/{participant_id}_training.npz (from FeatureStore)
    or feature_dir/{participant_id}_fused.npy (manually created).

    Args:
        split_csv: Path to split CSV
        feature_dir: Directory containing feature files
        id_column: Column name for participant ID

    Returns:
        {participant_id: np.ndarray (num_chunks, feature_dim)}
    """
    df = pd.read_csv(split_csv)
    features = {}

    for _, row in df.iterrows():
        pid = str(int(row[id_column]))

        # Try multiple naming conventions:
        #   1. {pid}_training.npz (FeatureStore output from extract_features.py)
        #   2. {pid}_fused.npy (pre-computed PCA-reduced)
        #   3. {pid}_features.npz (legacy naming)
        npz_path = feature_dir / f"{pid}_training.npz"
        npy_path = feature_dir / f"{pid}_fused.npy"
        legacy_path = feature_dir / f"{pid}_features.npz"

        if npz_path.exists():
            data = np.load(npz_path)
            if 'fused' in data:
                # Pre-computed PCA-reduced features
                features[pid] = data['fused']
            elif 'egemaps' in data and 'mfcc' in data:
                # Raw FeatureStore format — concatenate for caller to handle
                text_key = 'text_embeddings' if 'text_embeddings' in data else 'text'
                features[pid] = np.concatenate([
                    data['egemaps'], data['mfcc'], data[text_key]
                ], axis=1)
                logging.getLogger(__name__).debug(
                    f"[DATA_FLOW] {pid}: loaded raw npz, concatenated to {features[pid].shape}"
                )
            else:
                # Single array stored directly
                keys = list(data.files)
                features[pid] = data[keys[0]]
        elif npy_path.exists():
            features[pid] = np.load(npy_path)
        elif legacy_path.exists():
            data = np.load(legacy_path)
            if 'fused' in data:
                features[pid] = data['fused']
            elif 'egemaps' in data and 'mfcc' in data:
                text_key = 'text_embeddings' if 'text_embeddings' in data else 'text'
                features[pid] = np.concatenate([
                    data['egemaps'], data['mfcc'], data[text_key]
                ], axis=1)
            else:
                keys = list(data.files)
                features[pid] = data[keys[0]]
        else:
            logging.getLogger(__name__).warning(
                f"[VALIDATION_CHECK] Feature file missing for {pid} in {feature_dir}"
            )

    return features


def build_features_from_csvs(
    train_csv: Path,
    dev_csv: Path,
    feature_dir: Path,
    normalizer_path: Path,
    pca_path: Path,
    id_column: str = "Participant_ID",
    n_components: int = 24,
    feature_mode: str = "all",
) -> dict:
    """
    Load raw per-subject features and process them.

    feature_mode:
        "all" — normalize eGeMAPS+MFCC+text, fuse (592), PCA-reduce (n_components)
        "text_only" — L2-normalize text only (384 dims per chunk), skip PCA

    Returns:
        {split_name: {participant_id: np.ndarray (num_chunks, D)}}
    """
    logger = logging.getLogger(__name__)

    # Load all participant raw features
    splits = {'train': train_csv, 'dev': dev_csv}
    raw_features = {}

    for split_name, csv_path in splits.items():
        df = pd.read_csv(csv_path)
        raw_features[split_name] = {}
        for _, row in df.iterrows():
            pid = str(int(row[id_column]))
            # Try FeatureStore naming ({pid}_training.npz), then legacy ({pid}_features.npz)
            npz_path = feature_dir / f"{pid}_training.npz"
            if not npz_path.exists():
                npz_path = feature_dir / f"{pid}_features.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                raw_features[split_name][pid] = {
                    'egemaps': data['egemaps'],
                    'mfcc': data['mfcc'],
                    'text': data['text_embeddings'] if 'text_embeddings' in data else data['text'],
                }
            else:
                logger.warning(
                    f"[VALIDATION_CHECK] Missing features for {pid}. "
                    f"Run: python scripts/extract_features.py first."
                )

    if not raw_features['train']:
        raise RuntimeError("No training features found. Run feature extraction first.")

    # === TEXT-ONLY MODE: L2-normalize text, skip everything else ===
    if feature_mode == "text_only":
        result = {}
        for split_name, participants in raw_features.items():
            result[split_name] = {}
            for pid, feats in participants.items():
                text = feats['text'].astype(np.float32)
                # L2 normalize each chunk's text embedding
                norms = np.linalg.norm(text, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                result[split_name][pid] = text / norms
        logger.info(
            f"[TRAINING_PATH] Text-only mode: {len(result['train'])} train, "
            f"{len(result['dev'])} dev subjects, dim=384 per chunk"
        )
        # Create scalers directory (inference compatibility)
        Path(normalizer_path).parent.mkdir(parents=True, exist_ok=True)
        return result

    # === FULL MODE: normalize eGeMAPS+MFCC+text, fuse, PCA ===

    # --- Fit normalizer on train ---
    train_egemaps = np.concatenate(
        [f['egemaps'] for f in raw_features['train'].values()]
    )
    train_mfcc = np.concatenate(
        [f['mfcc'] for f in raw_features['train'].values()]
    )

    normalizer = FeatureNormalizer()
    normalizer.fit(train_egemaps, train_mfcc)

    # --- Normalize + Fuse all splits ---
    fusion = FeatureFusion()
    fused_per_split = {}

    for split_name, participants in raw_features.items():
        fused_per_split[split_name] = {}
        for pid, feats in participants.items():
            normed = normalizer.transform(
                feats['egemaps'], feats['mfcc'], feats['text']
            )
            fused = fusion.fuse(normed)
            fused_per_split[split_name][pid] = fused

    # --- Fit PCA on train, transform all ---
    train_all_fused = np.concatenate(
        list(fused_per_split['train'].values())
    )
    pca = PCAReducer(n_components=n_components)
    pca.fit(train_all_fused)

    reduced_per_split = {}
    for split_name, participants in fused_per_split.items():
        reduced_per_split[split_name] = {}
        for pid, fused in participants.items():
            reduced_per_split[split_name][pid] = pca.transform(fused)

    # --- Save normalizer + PCA for inference ---
    normalizer.save(str(normalizer_path))
    pca.save(str(pca_path))
    logger.info(
        f"[TRAINING_PATH] Saved normalizer to {normalizer_path}, PCA to {pca_path}"
    )

    return reduced_per_split


def main():
    parser = argparse.ArgumentParser(description="Train DepressionModel")
    parser.add_argument(
        "--config", type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--feature-dir", type=str,
        default="data/features",
        help="Directory with pre-extracted per-subject feature files",
    )
    parser.add_argument(
        "--use-precomputed-pca", action="store_true",
        help="If set, load pre-fused PCA-reduced .npy files from feature-dir",
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)
    log_cfg = config.get('logging', {})
    log_path = setup_logging(
        task_name="training",
        log_dir=log_cfg.get('log_dir', 'logs'),
        console_level="INFO",
        file_level="DEBUG",
    )
    logger = logging.getLogger(__name__)

    # --- GPU optimizations ---
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # TF32 for faster matmul on Ampere+ GPUs (RTX 30xx, A100, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Note: cudnn.benchmark is NOT set here — Trainer._set_seed() sets
        # deterministic=True and benchmark=False for reproducibility (MED-3).
        logger.info(f"[GPU] Using {torch.cuda.get_device_name(device)}, TF32 enabled, deterministic mode")
    else:
        logger.info("[GPU] No CUDA device found, running on CPU")

    # --- Run manager ---
    ckpt_cfg = config.get('checkpointing', {})
    run_mgr = RunManager(
        checkpoint_dir=ckpt_cfg.get('save_dir', 'checkpoints'),
        log_dir=log_cfg.get('log_dir', 'logs'),
    )
    run_mgr.save_config_snapshot(config)

    # --- Experiment tracker (DB) ---
    try:
        tracker = ExperimentTracker()
        logger.info("[DB] Experiment tracker initialized")
    except Exception as e:
        logger.warning(f"[DB] Tracker init failed (metrics won't be persisted): {e}")
        tracker = None

    logger.info("=" * 60)
    logger.info("[SESSION 7] Depression Severity Training Pipeline")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 60)

    data_cfg = config.get('data', {})
    train_csv = Path(data_cfg.get('train_csv', 'data/splits/train_split.csv'))
    dev_csv = Path(data_cfg.get('dev_csv', 'data/splits/dev_split.csv'))
    feature_dir = Path(args.feature_dir)
    ckpt_cfg = config.get('checkpointing', {})
    save_dir = Path(ckpt_cfg.get('save_dir', 'checkpoints'))

    # --- Load or build features ---
    model_cfg = config.get('model', {}) or {}
    if not model_cfg:
        model_config_path = Path("configs/model_config.yaml")
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                model_cfg = yaml.safe_load(f).get('model', {})
    expected_dim = model_cfg.get('input_dim', 64)

    if args.use_precomputed_pca:
        # Load pre-reduced features (already normalized + fused + PCA)
        logger.info("[TRAINING_PATH] Loading pre-computed PCA-reduced features")
        participant_features = {}
        for csv_path in [train_csv, dev_csv]:
            feats = load_participant_features(
                csv_path, feature_dir,
                id_column=data_cfg.get('id_column', 'Participant_ID'),
            )
            participant_features.update(feats)

        # Verify loaded features match model input_dim
        if participant_features:
            sample_dim = next(iter(participant_features.values())).shape[1]
            if sample_dim != expected_dim:
                logger.warning(
                    f"[VALIDATION_CHECK] --use-precomputed-pca: loaded features are "
                    f"{sample_dim}-dim but model expects {expected_dim}-dim. "
                    f"Falling back to full normalize → fuse → PCA pipeline."
                )
                args.use_precomputed_pca = False  # fall through to default path

    if not args.use_precomputed_pca:
        # Full pipeline: normalize → fuse → PCA
        logger.info("[TRAINING_PATH] Building features: normalize → fuse → PCA")
        normalizer_path = save_dir / "scalers" / "feature_scalers.pkl"
        pca_path = save_dir / "scalers" / "pca_reducer.pkl"

        reduced = build_features_from_csvs(
            train_csv=train_csv,
            dev_csv=dev_csv,
            feature_dir=feature_dir,
            normalizer_path=normalizer_path,
            pca_path=pca_path,
            id_column=data_cfg.get('id_column', 'Participant_ID'),
            n_components=expected_dim,
            feature_mode=model_cfg.get('feature_mode', 'all'),
        )
        # Merge all splits into one dict for SequenceBuilder
        participant_features = {}
        for split_feats in reduced.values():
            participant_features.update(split_feats)

    logger.info(
        f"[TRAINING_PATH] Loaded features for {len(participant_features)} participants"
    )

    # --- Build DataLoaders ---
    aug_cfg = config.get('augmentation', {})
    seq_builder = SequenceBuilder(
        batch_size=data_cfg.get('batch_size', 8),
        max_chunks=data_cfg.get('max_chunks', 0),
        num_workers=data_cfg.get('num_workers', 0),
        phq_threshold=data_cfg.get('phq_threshold', 10.0),
        temporal_dropout_rate=aug_cfg.get('temporal_dropout_rate', 0.05),
        feature_noise_std=aug_cfg.get('feature_noise_std', 0.005),
        label_noise_std=aug_cfg.get('label_noise_std', 0.15),
        depression_threshold=aug_cfg.get('depression_threshold', 10.0),
        detailed_labels_csv=config.get('detailed_labels_csv'),
    )

    loaders = seq_builder.build_train_loaders(
        participant_features=participant_features,
        train_csv=str(train_csv),
        dev_csv=str(dev_csv),
        label_column=data_cfg.get('label_column', 'PHQ_Score'),
        id_column=data_cfg.get('id_column', 'Participant_ID'),
    )

    # --- Build Model (model_cfg already loaded above for dim check) ---
    head_dropout = model_cfg.get('head', {}).get('dropout', 0.5)
    gru_dropout = model_cfg.get('bigru', {}).get('dropout', head_dropout)
    model = DepressionModel(
        input_dim=model_cfg.get('input_dim', 64),
        mlp_hidden=model_cfg.get('mlp', {}).get('hidden_dim', 15),
        mlp_dropout=model_cfg.get('mlp', {}).get('dropout', 0.5),
        mlp_bottleneck=model_cfg.get('mlp', {}).get('bottleneck', 20),
        gru_hidden=model_cfg.get('bigru', {}).get('hidden_size', 3),
        gru_layers=model_cfg.get('bigru', {}).get('num_layers', 1),
        head_dropout=head_dropout,
        gru_dropout=gru_dropout,
        pooling=model_cfg.get('pooling', 'stats_direct'),
        stats_head_dim=model_cfg.get('stats_head_dim', 0),
        stats_mode=model_cfg.get('stats_mode', 'mean_std'),
        multitask=model_cfg.get('multitask', False),
    )

    # --- Train ---
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        dev_loader=loaders['dev'],
        config=config,
        device=device,
    )

    import time as _time
    _train_start = _time.perf_counter()

    # Start experiment in DB
    exp_id = None
    if tracker:
        try:
            exp_id = tracker.start_experiment(
                config=config,
                train_samples=len(loaders['train'].dataset),
                dev_samples=len(loaders['dev'].dataset),
                max_epochs=config.get('training', {}).get('epochs', 100),
            )
        except Exception as e:
            logger.warning(f"[DB] Failed to start experiment: {e}")

    best_metrics = trainer.train()
    _train_elapsed = _time.perf_counter() - _train_start

    # Log per-epoch curves to DB
    if tracker and exp_id and hasattr(trainer, 'history'):
        try:
            num_epochs = len(trainer.history.get('train_loss', []))
            for i in range(num_epochs):
                tracker.log_epoch(
                    experiment_id=exp_id,
                    epoch=i + 1,
                    train_loss=trainer.history['train_loss'][i],
                    val_loss=trainer.history['val_loss'][i] if i < len(trainer.history.get('val_loss', [])) else None,
                    val_ccc=trainer.history['val_ccc'][i] if i < len(trainer.history.get('val_ccc', [])) else None,
                    val_rmse=trainer.history['val_rmse'][i] if i < len(trainer.history.get('val_rmse', [])) else None,
                    val_mae=trainer.history['val_mae'][i] if i < len(trainer.history.get('val_mae', [])) else None,
                    learning_rate=trainer.history['lr'][i] if i < len(trainer.history.get('lr', [])) else None,
                )
            logger.info(f"[DB] Logged {num_epochs} epoch curves")
        except Exception as e:
            logger.warning(f"[DB] Failed to log epoch curves: {e}")

    # Finish experiment in DB
    if tracker and exp_id:
        try:
            tracker.finish_experiment(
                experiment_id=exp_id,
                actual_epochs=trainer.early_stopping.best_epoch + 1,
                training_time_seconds=_train_elapsed,
                best_epoch=best_metrics.get('epoch', 0),
                best_ccc=best_metrics.get('ccc', 0.0),
                best_rmse=best_metrics.get('rmse', 0.0),
                best_mae=best_metrics.get('mae', 0.0),
                final_lr=trainer.optimizer.param_groups[0]['lr'],
                stopped_reason=(
                    'early_stopping' if trainer.early_stopping.should_stop
                    else 'max_epochs'
                ),
            )
        except Exception as e:
            logger.warning(f"[DB] Failed to finish experiment: {e}")

    # --- Save training summary ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_mgr.save_training_summary(
        best_metrics=best_metrics,
        total_epochs=trainer.early_stopping.best_epoch + 1,
        training_time_seconds=_train_elapsed,
        model_params=total_params,
        train_samples=len(loaders['train'].dataset),
        dev_samples=len(loaders['dev'].dataset),
    )
    run_mgr.print_artifact_status()

    # --- Report ---
    logger.info("=" * 60)
    logger.info("[SESSION 7] Training Complete")
    logger.info(f"  Best CCC:  {best_metrics['ccc']:.4f}")
    logger.info(f"  Best RMSE: {best_metrics['rmse']:.2f}")
    logger.info(f"  Best MAE:  {best_metrics['mae']:.2f}")
    logger.info(f"  Best Epoch: {best_metrics.get('epoch', '?')}")
    logger.info(f"  Training time: {_train_elapsed:.1f}s")
    logger.info("")
    logger.info("  To run inference:")
    logger.info("    python scripts/predict.py --audio <audio_file.wav>")
    logger.info("    python scripts/predict.py --audio-dir <audio_dir/> --output results.csv")
    logger.info("=" * 60)

    return best_metrics


if __name__ == "__main__":
    main()
