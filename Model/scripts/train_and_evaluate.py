#!/usr/bin/env python3
"""
Train the model and evaluate on train, validation, and test splits.
Reports CCC, MSE, MAE for all three splits.

Each run creates a timestamped folder under logs/runs/ containing:
  - run_summary.txt   (human-readable metrics + config)
  - metrics.json      (machine-readable results)
  - config_snapshot.yaml (exact config used)
  - training_log.log  (full training log)

Usage:
    python scripts/train_and_evaluate.py
    python scripts/train_and_evaluate.py --config configs/training_config.yaml
"""

import sys
import logging
import yaml
import json
import time
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.normalizer import FeatureNormalizer
from src.features.fusion import FeatureFusion
from src.features.pca_reducer import PCAReducer
from src.dataset import SequenceBuilder
from src.models import DepressionModel
from src.training import Trainer
from src.training.metrics import compute_all_metrics
from src.utils import setup_logging, RunManager


def create_run_dir() -> Path:
    """Create a timestamped run directory under logs/runs/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs/runs") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_summary(
    run_dir: Path,
    config: dict,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    best_epoch: int,
    train_time: float,
    total_params: int,
    test_pids: list,
    test_preds: np.ndarray,
    test_targets: np.ndarray,
):
    """Save a complete summary of the run to the run directory."""
    # 1. Config snapshot
    with open(run_dir / "config_snapshot.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # 2. Machine-readable metrics
    results = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'best_epoch': best_epoch,
        'training_time_s': train_time,
        'total_params': total_params,
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    # 3. Human-readable summary
    model_cfg = config.get('model', {})
    opt_cfg = config.get('optimizer', {})
    loss_cfg = config.get('loss', {})
    lines = [
        "=" * 70,
        "  RUN SUMMARY",
        "=" * 70,
        f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Training time: {train_time:.1f}s | Best epoch: {best_epoch}",
        f"  Parameters:   {total_params:,}",
        "",
        "  --- Config ---",
        f"  feature_mode:  {model_cfg.get('feature_mode', 'N/A')}",
        f"  stats_mode:    {model_cfg.get('stats_mode', 'N/A')}",
        f"  stats_head_dim:{model_cfg.get('stats_head_dim', 'N/A')}",
        f"  head_dropout:  {model_cfg.get('head', {}).get('dropout', 'N/A')}",
        f"  batch_size:    {config.get('data', {}).get('batch_size', 'N/A')}",
        f"  lr:            {opt_cfg.get('lr', 'N/A')}",
        f"  weight_decay:  {opt_cfg.get('weight_decay', 'N/A')}",
        f"  loss_type:     {loss_cfg.get('type', 'N/A')}",
        f"  ccc_weight:    {loss_cfg.get('ccc_weight', 'N/A')}",
        f"  seed:          {config.get('training', {}).get('seed', 'N/A')}",
        "",
        "  --- Metrics ---",
        f"  {'Split':<12} {'CCC':>10} {'MSE':>10} {'MAE':>10} {'RMSE':>10}",
        "-" * 70,
    ]
    for name, m in [("Train", train_metrics), ("Validation", val_metrics), ("Test", test_metrics)]:
        lines.append(f"  {name:<12} {m['ccc']:>10.4f} {m['mse']:>10.4f} {m['mae']:>10.4f} {m['rmse']:>10.4f}")
    lines.append("=" * 70)

    # Per-subject test predictions
    if len(test_pids) == len(test_preds):
        lines.append("")
        lines.append(f"  {'Subject':<12} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
        lines.append("-" * 50)
        for pid, pred, actual in zip(test_pids, test_preds, test_targets):
            lines.append(f"  {pid:<12} {pred:>10.2f} {actual:>10.1f} {abs(pred - actual):>10.2f}")
        lines.append("=" * 70)

    summary_text = "\n".join(lines)
    with open(run_dir / "run_summary.txt", 'w') as f:
        f.write(summary_text)

    return summary_text


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_features(split_csv: Path, feature_dir: Path, id_column: str = "Participant_ID") -> dict:
    """Load raw per-subject features from NPZ files."""
    df = pd.read_csv(split_csv)
    raw = {}
    for _, row in df.iterrows():
        pid = str(int(row[id_column]))
        npz_path = feature_dir / f"{pid}_training.npz"
        if not npz_path.exists():
            npz_path = feature_dir / f"{pid}_features.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            raw[pid] = {
                'egemaps': data['egemaps'],
                'mfcc': data['mfcc'],
                'text': data['text_embeddings'] if 'text_embeddings' in data else data['text'],
            }
        else:
            logging.getLogger(__name__).warning(f"Missing features for {pid}")
    return raw


def build_text_only_features(raw_features: dict) -> dict:
    """L2-normalize text embeddings only."""
    result = {}
    for pid, feats in raw_features.items():
        text = feats['text'].astype(np.float32)
        norms = np.linalg.norm(text, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        result[pid] = text / norms
    return result


@torch.no_grad()
def evaluate_split(model, loader, device):
    """Evaluate model on a DataLoader, return CCC, MSE, MAE."""
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        features = batch['features'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        lengths = batch['lengths']

        predictions = model(features, mask, lengths)
        all_preds.append(predictions.clamp(0.0, 24.0).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_all_metrics(all_preds, all_targets)
    # Add MSE explicitly
    metrics['mse'] = float(np.mean((all_preds - all_targets) ** 2))
    return metrics, all_preds, all_targets


def main():
    config = load_config("configs/training_config.yaml")

    # Create run directory
    run_dir = create_run_dir()

    setup_logging(task_name="train_eval", log_dir=str(run_dir), console_level="INFO", file_level="DEBUG")
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"[GPU] Using {torch.cuda.get_device_name(device)}")
    else:
        logger.info("[GPU] Running on CPU")

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {}) or {}
    if not model_cfg:
        model_config_path = Path("configs/model_config.yaml")
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                model_cfg = yaml.safe_load(f).get('model', {})

    feature_dir = Path("data/features")
    train_csv = Path(data_cfg.get('train_csv', 'data/splits/train_split.csv'))
    dev_csv = Path(data_cfg.get('dev_csv', 'data/splits/dev_split.csv'))
    test_csv = Path("data/splits/test_split.csv")

    feature_mode = model_cfg.get('feature_mode', 'all')

    # ── Load features for all 3 splits ──
    logger.info("Loading features for train/dev/test splits...")
    raw_train = load_raw_features(train_csv, feature_dir)
    raw_dev = load_raw_features(dev_csv, feature_dir)
    raw_test = load_raw_features(test_csv, feature_dir)

    if feature_mode == "text_only":
        train_feats = build_text_only_features(raw_train)
        dev_feats = build_text_only_features(raw_dev)
        test_feats = build_text_only_features(raw_test)
    else:
        # Full pipeline: normalize → fuse → PCA
        train_egemaps = np.concatenate([f['egemaps'] for f in raw_train.values()])
        train_mfcc = np.concatenate([f['mfcc'] for f in raw_train.values()])

        normalizer = FeatureNormalizer()
        normalizer.fit(train_egemaps, train_mfcc)

        fusion = FeatureFusion()
        n_components = model_cfg.get('input_dim', 24)

        all_raw = {'train': raw_train, 'dev': raw_dev, 'test': raw_test}
        fused_all = {}
        for split_name, participants in all_raw.items():
            fused_all[split_name] = {}
            for pid, feats in participants.items():
                normed = normalizer.transform(feats['egemaps'], feats['mfcc'], feats['text'])
                fused_all[split_name][pid] = fusion.fuse(normed)

        pca = PCAReducer(n_components=n_components)
        pca.fit(np.concatenate(list(fused_all['train'].values())))

        train_feats = {pid: pca.transform(f) for pid, f in fused_all['train'].items()}
        dev_feats = {pid: pca.transform(f) for pid, f in fused_all['dev'].items()}
        test_feats = {pid: pca.transform(f) for pid, f in fused_all['test'].items()}

        # Save scalers
        save_dir = Path(config.get('checkpointing', {}).get('save_dir', 'checkpoints'))
        scaler_dir = save_dir / "scalers"
        scaler_dir.mkdir(parents=True, exist_ok=True)
        normalizer.save(str(scaler_dir / "feature_scalers.pkl"))
        pca.save(str(scaler_dir / "pca_reducer.pkl"))

    logger.info(f"Features loaded: train={len(train_feats)}, dev={len(dev_feats)}, test={len(test_feats)}")

    # ── Build DataLoaders for train and dev (used in training) ──
    participant_features = {}
    participant_features.update(train_feats)
    participant_features.update(dev_feats)

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

    # ── Build test DataLoader ──
    # For test, we need a separate loader
    test_participant_features = {}
    test_participant_features.update(test_feats)

    test_seq_builder = SequenceBuilder(
        batch_size=data_cfg.get('batch_size', 8),
        max_chunks=data_cfg.get('max_chunks', 0),
        num_workers=data_cfg.get('num_workers', 0),
        phq_threshold=data_cfg.get('phq_threshold', 10.0),
        temporal_dropout_rate=0.0,  # No augmentation for test
        feature_noise_std=0.0,
        label_noise_std=0.0,
        depression_threshold=aug_cfg.get('depression_threshold', 10.0),
    )

    test_loaders = test_seq_builder.build_train_loaders(
        participant_features=test_participant_features,
        train_csv=str(test_csv),  # use test as "train" to load it
        dev_csv=str(test_csv),    # same for dev slot
        label_column=data_cfg.get('label_column', 'PHQ_Score'),
        id_column=data_cfg.get('id_column', 'Participant_ID'),
    )
    test_loader = test_loaders['train']  # the test set loaded as a DataLoader

    # ── Build Model ──
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

    # ── Train ──
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        dev_loader=loaders['dev'],
        config=config,
        device=device,
    )

    t0 = time.perf_counter()
    best_val_metrics = trainer.train()
    train_time = time.perf_counter() - t0

    # ── Load best checkpoint ──
    ckpt_cfg = config.get('checkpointing', {})
    ckpt_path = Path(ckpt_cfg.get('save_dir', 'checkpoints')) / ckpt_cfg.get('model_filename', 'best_model.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    logger.info(f"Loaded best checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # ── Evaluate on all splits (with best model, no augmentation) ──
    # Rebuild train loader without augmentation for fair eval
    eval_seq_builder = SequenceBuilder(
        batch_size=data_cfg.get('batch_size', 8),
        max_chunks=data_cfg.get('max_chunks', 0),
        num_workers=data_cfg.get('num_workers', 0),
        phq_threshold=data_cfg.get('phq_threshold', 10.0),
        temporal_dropout_rate=0.0,
        feature_noise_std=0.0,
        label_noise_std=0.0,
        depression_threshold=aug_cfg.get('depression_threshold', 10.0),
    )
    eval_loaders = eval_seq_builder.build_train_loaders(
        participant_features=participant_features,
        train_csv=str(train_csv),
        dev_csv=str(dev_csv),
        label_column=data_cfg.get('label_column', 'PHQ_Score'),
        id_column=data_cfg.get('id_column', 'Participant_ID'),
    )

    train_metrics, train_preds, train_targets = evaluate_split(model, eval_loaders['train'], device)
    val_metrics, val_preds, val_targets = evaluate_split(model, eval_loaders['dev'], device)
    test_metrics, test_preds, test_targets = evaluate_split(model, test_loader, device)

    # ── Print final report ──
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 70)
    print("  FINAL MODEL METRICS (Best Checkpoint)")
    print("=" * 70)
    print(f"  Training time: {train_time:.1f}s | Best epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Total trainable parameters: {total_params:,}")
    print("-" * 70)
    print(f"  {'Split':<12} {'CCC':>10} {'MSE':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 70)
    for name, m in [("Train", train_metrics), ("Validation", val_metrics), ("Test", test_metrics)]:
        print(f"  {name:<12} {m['ccc']:>10.4f} {m['mse']:>10.4f} {m['mae']:>10.4f} {m['rmse']:>10.4f}")
    print("=" * 70)

    # ── Per-subject test results ──
    test_df = pd.read_csv(test_csv)
    test_pids = []
    for _, row in test_df.iterrows():
        pid = str(int(row['Participant_ID']))
        if pid in test_feats:
            test_pids.append(pid)

    if len(test_pids) == len(test_preds):
        print(f"\n  {'Subject':<12} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
        print("-" * 50)
        for pid, pred, actual in zip(test_pids, test_preds, test_targets):
            print(f"  {pid:<12} {pred:>10.2f} {actual:>10.1f} {abs(pred - actual):>10.2f}")
        print("=" * 70)

    # ── Save to run directory ──
    summary_text = save_run_summary(
        run_dir=run_dir,
        config=config,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_epoch=checkpoint.get('epoch', 0),
        train_time=train_time,
        total_params=total_params,
        test_pids=test_pids,
        test_preds=test_preds,
        test_targets=test_targets,
    )

    # Also save to legacy location for backward compat
    results = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'best_epoch': checkpoint.get('epoch', 0),
        'training_time_s': train_time,
    }
    results_path = Path("logs/final_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Results saved to {results_path}")
    print(f"\n  Run saved to: {run_dir}")

    return results


if __name__ == "__main__":
    main()
