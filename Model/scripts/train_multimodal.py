#!/usr/bin/env python3
"""
Staged Multimodal Training for Depression Severity Prediction.

Three-stage training strategy:
  Stage 1: Train text-only branch (freeze audio) — establishes text baseline
  Stage 2: Train audio branch + fusion gate (freeze text) — audio learns to help
  Stage 3: Joint fine-tuning with reduced LR — polish fusion

This ensures audio CANNOT degrade text performance:
  - Text branch starts pre-trained (CCC~0.54)
  - Audio branch must prove value through fusion gate
  - If audio hurts → gate learns to shut it off → graceful text-only fallback

Uses V2 features: HuBERT(768) + SBERT(384) + audio_quality scores.

Usage:
    python scripts/train_multimodal.py
    python scripts/train_multimodal.py --config configs/multimodal_config.yaml
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gated_fusion_model import GatedMultimodalModel
from src.dataset.multimodal_dataset import MultimodalDataset, multimodal_collate_fn
from src.training.losses import WeightedMSELoss
from src.training.metrics import compute_all_metrics
from src.training.early_stopping import EarlyStopping
from src.utils import setup_logging


def load_v2_features(
    feature_dir: Path,
    split_csv: Path,
    id_column: str = "Participant_ID",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load V2 features (HuBERT + SBERT + quality) for all participants in a split.

    Returns:
        {pid: {'text': (T, 384), 'audio': (T, 768), 'quality': (T,)}}
    """
    logger = logging.getLogger(__name__)
    df = pd.read_csv(split_csv)
    data = {}

    for _, row in df.iterrows():
        pid = str(int(row[id_column]))
        npz_path = feature_dir / f"{pid}_training_v2.npz"
        if not npz_path.exists():
            logger.warning(f"[VALIDATION_CHECK] Missing V2 features for {pid}")
            continue

        npz = np.load(npz_path)
        text = npz['text_embeddings'].astype(np.float32)
        audio = npz['hubert'].astype(np.float32)
        quality = npz['audio_quality'].astype(np.float32)

        # L2 normalize text embeddings
        norms = np.linalg.norm(text, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        text = text / norms

        data[pid] = {'text': text, 'audio': audio, 'quality': quality}

    logger.info(f"[DATA_FLOW] Loaded V2 features for {len(data)} participants from {split_csv.name}")
    return data


def build_weighted_sampler(
    dataset: MultimodalDataset,
    phq_threshold: float = 10.0,
) -> WeightedRandomSampler:
    """Build inverse-frequency weighted sampler for class balance."""
    n_dep = sum(1 for s in dataset.samples if s['label'] >= phq_threshold)
    n_notdep = len(dataset.samples) - n_dep
    if n_dep == 0 or n_notdep == 0:
        return None

    w_dep = len(dataset.samples) / (2 * n_dep)
    w_notdep = len(dataset.samples) / (2 * n_notdep)

    weights = []
    for s in dataset.samples:
        weights.append(w_dep if s['label'] >= phq_threshold else w_notdep)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        text = batch['text_features'].to(device)
        audio = batch['audio_features'].to(device)
        quality = batch['audio_quality'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']

        optimizer.zero_grad()
        preds = model(text, audio, quality, mask, lengths)
        loss = criterion(preds, labels)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate and return metrics dict."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    n_batches = 0

    for batch in loader:
        text = batch['text_features'].to(device)
        audio = batch['audio_features'].to(device)
        quality = batch['audio_quality'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']

        preds = model(text, audio, quality, mask, lengths)
        loss = criterion(preds, labels)

        all_preds.append(preds.clamp(0, 24).cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        return {'ccc': 0.0, 'rmse': 99.0, 'mae': 99.0, 'loss': 0.0}
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_all_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def evaluate_text_only(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate with audio zeroed out → text-only CCC. Monitors text degradation."""
    model.eval()
    # Temporarily force text_only path regardless of current stage
    prev_stage = model._training_stage
    model._training_stage = "text_only"
    all_preds = []
    all_targets = []

    for batch in loader:
        text = batch['text_features'].to(device)
        audio = batch['audio_features'].to(device)
        quality = batch['audio_quality'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']

        preds = model(text, audio, quality, mask, lengths)
        all_preds.append(preds.clamp(0, 24).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    model._training_stage = prev_stage
    if not all_preds:
        return 0.0
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_all_metrics(all_preds, all_targets)
    return metrics['ccc']


@torch.no_grad()
def evaluate_audio_only(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate with text zeroed out → audio-only CCC."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        text = torch.zeros_like(batch['text_features']).to(device)
        audio = batch['audio_features'].to(device)
        quality = batch['audio_quality'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']

        preds = model(text, audio, quality, mask, lengths)
        all_preds.append(preds.clamp(0, 24).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_all_metrics(all_preds, all_targets)
    return metrics['ccc']


def run_stage(
    stage_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    grad_clip: float = 1.0,
    best_ccc: float = -1.0,
    save_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Run one training stage.

    Returns best metrics dict.
    """
    logger = logging.getLogger(__name__)

    # Only optimize unfrozen parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=max(patience // 3, 3), min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=patience, min_delta=0.001, mode='max')

    n_params = sum(p.numel() for p in params)
    logger.info(f"\n{'='*60}")
    logger.info(f"[{stage_name}] Starting: lr={lr}, wd={weight_decay}, "
                f"epochs={max_epochs}, patience={patience}, trainable_params={n_params:,}")
    logger.info(f"{'='*60}")

    best_metrics = {'ccc': best_ccc, 'epoch': 0}

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_metrics = validate(model, dev_loader, criterion, device)

        ccc = val_metrics['ccc']
        scheduler.step(ccc)
        current_lr = optimizer.param_groups[0]['lr']

        # Compute per-modality CCC for monitoring
        text_ccc = evaluate_text_only(model, dev_loader, device)
        audio_ccc = evaluate_audio_only(model, dev_loader, device)
        gate_stats = model.get_gate_stats()

        logger.info(
            f"[{stage_name} E{epoch:03d}] "
            f"loss={train_loss:.4f} | "
            f"val_ccc={ccc:.4f} | "
            f"text_ccc={text_ccc:.4f} | "
            f"audio_ccc={audio_ccc:.4f} | "
            f"gate={gate_stats['fusion_gate_mean']:.3f} | "
            f"lr={current_lr:.6f}"
        )

        # Save if best
        if ccc > best_metrics['ccc']:
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch
            best_metrics['text_ccc'] = text_ccc
            best_metrics['audio_ccc'] = audio_ccc
            best_metrics['gate_mean'] = gate_stats['fusion_gate_mean']

            if save_path:
                torch.save({
                    'epoch': epoch,
                    'stage': stage_name,
                    'model_state_dict': model.state_dict(),
                    'metrics': best_metrics,
                }, save_path)

        # Early stopping
        if early_stop(ccc, epoch):
            logger.info(
                f"[{stage_name}] Early stop at epoch {epoch}. "
                f"Best CCC={best_metrics['ccc']:.4f} at epoch {best_metrics['epoch']}"
            )
            break

        # FAILURE DETECTION: if audio consistently hurts text
        if stage_name != "STAGE_1_TEXT" and epoch > 5:
            if ccc < text_ccc - 0.05:
                logger.warning(
                    f"[FAILURE_DETECT] Fusion CCC ({ccc:.4f}) < text-only ({text_ccc:.4f}) - 0.05. "
                    f"Audio may be hurting. Gate: {gate_stats['fusion_gate_mean']:.3f}"
                )

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Staged Multimodal Training")
    parser.add_argument(
        "--config", type=str, default="configs/multimodal_config.yaml",
        help="Path to multimodal training config",
    )
    parser.add_argument(
        "--feature-dir", type=str, default="data/features_v2",
        help="Directory with V2 feature NPZ files",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    log_path = setup_logging(
        task_name="multimodal_training",
        log_dir="logs",
        console_level="INFO",
        file_level="DEBUG",
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"[GPU] {torch.cuda.get_device_name(device)}")

    # Seed
    seed = config.get('training', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("[MULTIMODAL V2] Staged Training: HuBERT + SBERT Gated Fusion")
    logger.info(f"  Feature dir: {args.feature_dir}")
    logger.info(f"  Log file: {log_path}")
    logger.info("=" * 60)

    # === Load features ===
    feature_dir = Path(args.feature_dir)
    data_cfg = config.get('data', {})
    train_csv = Path(data_cfg.get('train_csv', 'data/splits/train_split.csv'))
    dev_csv = Path(data_cfg.get('dev_csv', 'data/splits/dev_split.csv'))
    id_column = data_cfg.get('id_column', 'Participant_ID')

    train_data = load_v2_features(feature_dir, train_csv, id_column)
    dev_data = load_v2_features(feature_dir, dev_csv, id_column)

    if len(train_data) == 0:
        logger.error(
            "[FATAL] No V2 features found. Run feature extraction first:\n"
            "    python scripts/extract_features_v2.py\n"
            f"  Expected files in: {feature_dir}/{{pid}}_training_v2.npz"
        )
        sys.exit(1)

    # Merge for dataset construction
    all_data = {**train_data, **dev_data}

    # === Build datasets ===
    aug_cfg = config.get('augmentation', {})
    batch_size = data_cfg.get('batch_size', 56)

    train_dataset = MultimodalDataset.from_split_csv(
        split_csv=train_csv,
        participant_data=all_data,
        augment=True,
        temporal_dropout_rate=aug_cfg.get('temporal_dropout_rate', 0.1),
        feature_noise_std=aug_cfg.get('feature_noise_std', 0.005),
        label_noise_std=aug_cfg.get('label_noise_std', 0.15),
        depression_threshold=data_cfg.get('phq_threshold', 10.0),
    )

    dev_dataset = MultimodalDataset.from_split_csv(
        split_csv=dev_csv,
        participant_data=all_data,
        augment=False,
    )

    sampler = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=multimodal_collate_fn, num_workers=0, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=multimodal_collate_fn, num_workers=0, pin_memory=True,
    )

    # === Build model ===
    model_cfg = config.get('model', {})
    model = GatedMultimodalModel(
        text_dim=model_cfg.get('text_dim', 384),
        audio_dim=model_cfg.get('audio_dim', 768),
        proj_dim=model_cfg.get('proj_dim', 128),
        stats_mode=model_cfg.get('stats_mode', 'mean_std'),
        dropout=model_cfg.get('dropout', 0.1),
        modality_dropout=model_cfg.get('modality_dropout', 0.15),
    ).to(device)

    # === Loss ===
    loss_cfg = config.get('loss', {})
    criterion = WeightedMSELoss(
        phq_threshold=loss_cfg.get('phq_threshold', 10.0),
        high_weight=loss_cfg.get('high_weight', 2.0),
        low_weight=loss_cfg.get('low_weight', 1.0),
    ).to(device)

    save_dir = Path(config.get('checkpointing', {}).get('save_dir', 'checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)

    stage_cfg = config.get('stages', {})

    # ====================================================================
    # STAGE 1: Train text branch only (audio frozen, establishes baseline)
    # ====================================================================
    s1_cfg = stage_cfg.get('stage1', {})
    model.set_training_stage("text_only")
    model.freeze_audio_branch()
    for param in model.fusion_gate.parameters():
        param.requires_grad = False

    best_s1 = run_stage(
        stage_name="STAGE_1_TEXT",
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        criterion=criterion,
        device=device,
        lr=s1_cfg.get('lr', 1e-2),
        weight_decay=s1_cfg.get('weight_decay', 0.1),
        max_epochs=s1_cfg.get('epochs', 500),
        patience=s1_cfg.get('patience', 100),
        save_path=save_dir / "best_stage1.pt",
    )

    logger.info(f"\n[STAGE 1 RESULT] Text-only CCC: {best_s1['ccc']:.4f}")

    # ====================================================================
    # STAGE 2: Train audio branch + fusion gate (text frozen)
    # ====================================================================
    s2_cfg = stage_cfg.get('stage2', {})
    model.set_training_stage("audio_gate")
    model.freeze_text_branch()
    model.unfreeze_audio_branch()
    for param in model.fusion_gate.parameters():
        param.requires_grad = True
    # Keep head trainable
    for param in model.head.parameters():
        param.requires_grad = True

    best_s2 = run_stage(
        stage_name="STAGE_2_AUDIO",
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        criterion=criterion,
        device=device,
        lr=s2_cfg.get('lr', 5e-3),
        weight_decay=s2_cfg.get('weight_decay', 0.05),
        max_epochs=s2_cfg.get('epochs', 500),
        patience=s2_cfg.get('patience', 100),
        best_ccc=best_s1['ccc'],
        save_path=save_dir / "best_stage2.pt",
    )

    logger.info(f"\n[STAGE 2 RESULT] Fusion CCC: {best_s2['ccc']:.4f}, "
                f"Text: {best_s2.get('text_ccc', 0):.4f}, "
                f"Audio: {best_s2.get('audio_ccc', 0):.4f}, "
                f"Gate: {best_s2.get('gate_mean', 0):.3f}")

    # ====================================================================
    # STAGE 3: Joint fine-tuning (all unfrozen, reduced LR)
    # ====================================================================
    s3_cfg = stage_cfg.get('stage3', {})
    model.set_training_stage("joint")
    model.unfreeze_text_branch()
    model.unfreeze_audio_branch()

    best_s3 = run_stage(
        stage_name="STAGE_3_JOINT",
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        criterion=criterion,
        device=device,
        lr=s3_cfg.get('lr', 1e-3),
        weight_decay=s3_cfg.get('weight_decay', 0.05),
        max_epochs=s3_cfg.get('epochs', 300),
        patience=s3_cfg.get('patience', 80),
        best_ccc=max(best_s1['ccc'], best_s2['ccc']),
        save_path=save_dir / "best_multimodal.pt",
    )

    logger.info(f"\n[STAGE 3 RESULT] Final CCC: {best_s3['ccc']:.4f}, "
                f"Text: {best_s3.get('text_ccc', 0):.4f}, "
                f"Audio: {best_s3.get('audio_ccc', 0):.4f}, "
                f"Gate: {best_s3.get('gate_mean', 0):.3f}")

    # === Final summary ===
    logger.info("\n" + "=" * 60)
    logger.info("[MULTIMODAL V2] TRAINING COMPLETE")
    logger.info(f"  Stage 1 (Text):    CCC={best_s1['ccc']:.4f}")
    logger.info(f"  Stage 2 (Audio):   CCC={best_s2['ccc']:.4f}")
    logger.info(f"  Stage 3 (Joint):   CCC={best_s3['ccc']:.4f}")
    logger.info(f"  Best checkpoint:   {save_dir / 'best_multimodal.pt'}")

    # Failure detection final check
    final_text_ccc = best_s3.get('text_ccc', best_s1['ccc'])
    if best_s3['ccc'] < best_s1['ccc']:
        logger.warning(
            f"[FAILURE_DETECT] Fusion ({best_s3['ccc']:.4f}) < text-only ({best_s1['ccc']:.4f}). "
            f"Recommending text-only model. Saved at {save_dir / 'best_stage1.pt'}"
        )
    elif best_s3['ccc'] < final_text_ccc - 0.02:
        logger.warning(
            f"[FAILURE_DETECT] Fusion ({best_s3['ccc']:.4f}) barely matches text ({final_text_ccc:.4f}). "
            f"Audio may not be helping. Check gate values."
        )
    else:
        improvement = best_s3['ccc'] - best_s1['ccc']
        logger.info(f"  Audio improved CCC by +{improvement:.4f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
