#!/usr/bin/env python3
"""
5-Fold Cross-Validation Training Script.

Merges train+dev splits → 219 subjects, runs stratified 5-fold CV,
saves per-fold checkpoints and reports mean±std metrics.

Usage:
    python scripts/train_cv.py
    python scripts/train_cv.py --config configs/training_config.yaml
"""

import sys
import argparse
import json
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.model_selection import StratifiedKFold

from scripts.train import load_config, build_features_from_csvs
from src.dataset import SequenceBuilder
from src.dataset.depression_dataset import DepressionDataset
from src.dataset.collate import depression_collate_fn
from src.models import DepressionModel
from src.training import Trainer
from src.utils import setup_logging

from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="5-Fold CV Training")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--feature-dir", type=str, default="data/features")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(task_name="cv_training", log_dir=config.get('logging', {}).get('log_dir', 'logs'))
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    aug_cfg = config.get('augmentation', {})
    train_csv = Path(data_cfg.get('train_csv', 'data/splits/train_split.csv'))
    dev_csv = Path(data_cfg.get('dev_csv', 'data/splits/dev_split.csv'))
    feature_dir = Path(args.feature_dir)

    # Build features (full pipeline)
    ckpt_cfg = config.get('checkpointing', {})
    save_dir = Path(ckpt_cfg.get('save_dir', 'checkpoints'))
    normalizer_path = save_dir / "scalers" / "feature_scalers.pkl"
    pca_path = save_dir / "scalers" / "pca_reducer.pkl"

    reduced = build_features_from_csvs(
        train_csv=train_csv,
        dev_csv=dev_csv,
        feature_dir=feature_dir,
        normalizer_path=normalizer_path,
        pca_path=pca_path,
        id_column=data_cfg.get('id_column', 'Participant_ID'),
        n_components=model_cfg.get('input_dim', 384),
        feature_mode=model_cfg.get('feature_mode', 'text_only'),
    )

    # Merge all features
    all_features = {}
    for split_feats in reduced.values():
        all_features.update(split_feats)

    # Merge labels from both CSVs
    all_labels = {}
    for csv_path in [train_csv, dev_csv]:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            pid = str(int(row[data_cfg.get('id_column', 'Participant_ID')]))
            all_labels[pid] = float(row[data_cfg.get('label_column', 'PHQ_Score')])

    # Filter to subjects with both features and labels
    valid_pids = sorted(set(all_features.keys()) & set(all_labels.keys()))
    scores = np.array([all_labels[pid] for pid in valid_pids])

    logger.info(f"Total subjects for CV: {len(valid_pids)}, PHQ range: [{scores.min()}, {scores.max()}]")

    # Stratified bins: 0-4, 5-9, 10-14, 15-19, 20-24
    bins = np.digitize(scores, bins=[5, 10, 15, 20])

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    cv_dir = save_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    fold_splits = {}

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(valid_pids, bins)):
        fold_seed = args.seed + fold_idx
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx+1}/{args.n_folds} (seed={fold_seed})")
        logger.info(f"{'='*60}")

        train_pids = [valid_pids[i] for i in train_indices]
        val_pids = [valid_pids[i] for i in val_indices]
        fold_splits[f"fold_{fold_idx}"] = {"train": train_pids, "val": val_pids}

        logger.info(f"Train: {len(train_pids)}, Val: {len(val_pids)}")

        # Build datasets directly
        detailed_csv = config.get('detailed_labels_csv')
        train_labels = {pid: all_labels[pid] for pid in train_pids}
        val_labels = {pid: all_labels[pid] for pid in val_pids}

        # Load item labels if available
        item_labels = {}
        if detailed_csv and Path(detailed_csv).exists():
            detail_df = pd.read_csv(detailed_csv)
            item_cols = [c for c in detail_df.columns if c.startswith("PHQ_8") and c != "PHQ_8Total"]
            for _, row in detail_df.iterrows():
                pid = str(int(row["Participant_ID"]))
                item_labels[pid] = [float(row[c]) for c in item_cols]

        train_dataset = DepressionDataset(
            participant_features=all_features,
            labels=train_labels,
            participant_ids=train_pids,
            max_chunks=data_cfg.get('max_chunks', 0),
            augment=True,
            temporal_dropout_rate=aug_cfg.get('temporal_dropout_rate', 0.15),
            feature_noise_std=aug_cfg.get('feature_noise_std', 0.01),
            label_noise_std=aug_cfg.get('label_noise_std', 0.2),
            depression_threshold=aug_cfg.get('depression_threshold', 10.0),
            item_labels=item_labels,
        )

        val_dataset = DepressionDataset(
            participant_features=all_features,
            labels=val_labels,
            participant_ids=val_pids,
            max_chunks=data_cfg.get('max_chunks', 0),
            item_labels=item_labels,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(data_cfg.get('batch_size', 163), len(train_pids)),
            shuffle=True,
            collate_fn=depression_collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_pids),
            shuffle=False,
            collate_fn=depression_collate_fn,
            num_workers=0,
        )

        # Build model
        head_dropout = model_cfg.get('head', {}).get('dropout', 0.0)
        model = DepressionModel(
            input_dim=model_cfg.get('input_dim', 384),
            pooling=model_cfg.get('pooling', 'stats_direct'),
            stats_mode=model_cfg.get('stats_mode', 'mean_std'),
            stats_head_dim=model_cfg.get('stats_head_dim', 0),
            head_dropout=head_dropout,
            multitask=model_cfg.get('multitask', False),
        )

        # Override config for this fold
        fold_config = dict(config)
        fold_config['training'] = dict(config.get('training', {}))
        fold_config['training']['seed'] = fold_seed
        fold_config['checkpointing'] = dict(config.get('checkpointing', {}))
        fold_config['checkpointing']['save_dir'] = str(cv_dir)
        fold_config['checkpointing']['model_filename'] = f"best_model_fold{fold_idx}.pt"

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            dev_loader=val_loader,
            config=fold_config,
            device=device,
        )

        best = trainer.train()
        best['fold'] = fold_idx
        fold_metrics.append(best)

        logger.info(f"Fold {fold_idx+1} best: CCC={best['ccc']:.4f}, RMSE={best['rmse']:.2f}, MAE={best['mae']:.2f}")

    # Summary
    cccs = [m['ccc'] for m in fold_metrics]
    rmses = [m['rmse'] for m in fold_metrics]
    maes = [m['mae'] for m in fold_metrics]

    logger.info(f"\n{'='*60}")
    logger.info(f"CV RESULTS ({args.n_folds} folds)")
    logger.info(f"{'='*60}")
    logger.info(f"CCC:  {np.mean(cccs):.4f} ± {np.std(cccs):.4f}")
    logger.info(f"RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    logger.info(f"MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f}")

    # Save results
    results = {
        "n_folds": args.n_folds,
        "seed": args.seed,
        "per_fold": fold_metrics,
        "summary": {
            "ccc_mean": float(np.mean(cccs)),
            "ccc_std": float(np.std(cccs)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
        }
    }
    with open(cv_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(cv_dir / "fold_splits.json", "w") as f:
        json.dump(fold_splits, f, indent=2)

    logger.info(f"Results saved to {cv_dir}/cv_results.json")
    logger.info(f"Fold splits saved to {cv_dir}/fold_splits.json")


if __name__ == "__main__":
    main()
