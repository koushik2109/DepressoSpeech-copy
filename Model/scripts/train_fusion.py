#!/usr/bin/env python3
"""
Multimodal Fusion Training — DepressoSpeech

Trains a late-residual fusion model on aligned text + MFCC + eGeMAPS features.
Text branch is loaded from a pretrained checkpoint and frozen.
Audio branch learns a residual correction via a learnable gate.

Usage:
    python scripts/train_fusion.py
    python scripts/train_fusion.py --lr 0.005 --epochs 2000
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multimodal_fusion import MultimodalFusion
from src.models.statistics_pooling import StatisticsPooling
from src.training.metrics import concordance_correlation_coefficient as ccc_metric
from src.training.losses import WeightedMSELoss, CombinedLoss
from src.utils.log_manager import setup_logger


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient loss: 1 - CCC.
    Directly optimizes the evaluation metric."""
    def forward(self, pred, target):
        mean_p = pred.mean()
        mean_t = target.mean()
        var_p = pred.var(unbiased=False)
        var_t = target.var(unbiased=False)
        cov = ((pred - mean_p) * (target - mean_t)).mean()
        ccc = 2 * cov / (var_p + var_t + (mean_p - mean_t) ** 2 + 1e-8)
        return 1 - ccc


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral Feature Extraction (interview-level, from transcripts)
# ─────────────────────────────────────────────────────────────────────────────

def extract_behavioral(pid: str, data_dir: str = "data/raw") -> np.ndarray:
    """
    Extract 16 interview-level behavioral features from transcript CSV.

    Features capture clinically-meaningful depression indicators:
        Turn-taking:  n_turns, avg_duration, duration_std
        Pauses:       avg_pause, pause_std, median_pause, long_pause_fraction
        Verbosity:    avg_words, words_std, speaking_rate, rate_std
        Global:       total_speaking_time, speaking_ratio, interview_duration
        Extremes:     max_turn_duration, total_word_count

    Returns: (16,) float32 or None if transcript unavailable
    """
    csv_path = Path(data_dir) / pid / f"{pid}_Transcript.csv"
    if not csv_path.exists():
        return None

    rows = []
    with open(csv_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",", 3)
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[0]), float(parts[1]), parts[2].strip()))
                except ValueError:
                    continue

    if len(rows) < 2:
        return None

    starts = np.array([r[0] for r in rows])
    ends = np.array([r[1] for r in rows])
    texts = [r[2] for r in rows]

    durations = ends - starts
    gaps = starts[1:] - ends[:-1]
    word_counts = np.array([len(t.split()) for t in texts])
    speaking_rates = word_counts / np.maximum(durations, 0.1)

    total_speak = durations.sum()
    total_interview = ends[-1] - starts[0]

    return np.array([
        len(rows),
        durations.mean(),
        durations.std(),
        gaps.mean(),
        gaps.std(),
        np.median(gaps),
        (gaps > 3.0).mean(),
        word_counts.mean(),
        word_counts.std(),
        speaking_rates.mean(),
        speaking_rates.std(),
        total_speak,
        total_speak / max(total_interview, 1),
        total_interview,
        durations.max(),
        word_counts.sum(),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(csv_path: Path, id_col="Participant_ID", label_col="PHQ_Score"):
    """Load participant labels from split CSV."""
    df = pd.read_csv(csv_path)
    return {str(int(row[id_col])): float(row[label_col]) for _, row in df.iterrows()}


def load_split(csv_path: Path, feature_dir: Path, data_dir: str = "data/raw",
               id_col="Participant_ID"):
    """Load aligned features + behavioral + labels for one split."""
    labels = load_labels(csv_path)
    pids = list(labels.keys())

    data = []
    for pid in pids:
        npz_path = feature_dir / f"{pid}_training.npz"
        if not npz_path.exists():
            continue
        f = np.load(npz_path)
        text = f["text_embeddings"].astype(np.float32)   # (N, 384)
        mfcc = f["mfcc"].astype(np.float32)               # (N, 120)
        egemaps = f["egemaps"].astype(np.float32)          # (N, 88)

        # L2 normalize text embeddings (same as text-only pipeline)
        norms = np.linalg.norm(text, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        text = text / norms

        # Extract behavioral features from transcript
        behavioral = extract_behavioral(pid, data_dir)
        if behavioral is None:
            behavioral = np.zeros(16, dtype=np.float32)

        data.append({
            "pid": pid,
            "text": text,
            "mfcc": mfcc,
            "egemaps": egemaps,
            "behavioral": behavioral,
            "label": labels[pid],
            "length": text.shape[0],
        })

    return data


def collate_multimodal(batch):
    """Pad variable-length sequences, build masks, stack behavioral features."""
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    B = len(batch)
    T = max(b["length"] for b in batch)

    text = torch.zeros(B, T, 384)
    mfcc = torch.zeros(B, T, 120)
    egemaps = torch.zeros(B, T, 88)
    behavioral = torch.zeros(B, 16)
    labels = torch.zeros(B)
    mask = torch.zeros(B, T, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["length"]
        text[i, :n] = torch.from_numpy(b["text"][:n])
        mfcc[i, :n] = torch.from_numpy(b["mfcc"][:n])
        egemaps[i, :n] = torch.from_numpy(b["egemaps"][:n])
        behavioral[i] = torch.from_numpy(b["behavioral"])
        labels[i] = b["label"]
        mask[i, :n] = True

    return text, mfcc, egemaps, behavioral, labels, mask


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, data, device, batch_size=256):
    """Evaluate model on a dataset split. Returns (predictions, targets, metrics)."""
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for text, mfcc, egemaps, behavioral, labels, mask in loader:
            text = text.to(device)
            mfcc = mfcc.to(device)
            egemaps = egemaps.to(device)
            behavioral = behavioral.to(device)
            mask = mask.to(device)
            preds = model(text, mfcc, egemaps, mask, behavioral).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    ccc = ccc_metric(preds, targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))

    return preds, targets, {"ccc": ccc, "rmse": rmse, "mae": mae}


def train(args):
    log = logging.getLogger("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── Data ──
    feature_dir = Path(args.feature_dir)
    train_data = load_split(Path(args.train_csv), feature_dir, args.data_dir)
    dev_data = load_split(Path(args.dev_csv), feature_dir, args.data_dir)
    test_data = load_split(Path(args.test_csv), feature_dir, args.data_dir)

    train_labels = np.array([d["label"] for d in train_data])

    # ── Model ──
    model = MultimodalFusion(stats_mode="mean_std", residual_dropout=args.dropout).to(device)
    model.load_pretrained_text(args.checkpoint)

    params = model.param_summary()

    # ── Print Banner ──
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    sep = "═" * 72

    log.info(sep)
    log.info("  DepressoSpeech │ Multimodal Fusion Training")
    log.info(sep)
    log.info(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Device      : {gpu_name}")
    log.info(f"  Seed        : {args.seed}")
    log.info("")
    log.info("  MODEL")
    log.info(f"    Text      : StatsPool(384→768) → BN → Linear(768,1)  [FROZEN, pretrained]")
    log.info(f"    Audio     : StatsPool(208→416) ─┐")
    log.info(f"    Behavioral: (16) ───────────────┤→ cat(432) → BN → Linear(432,1)")
    log.info(f"    Fusion    : text_pred + residual                     [additive, init ≈ 0]")
    log.info(f"    Params    : {params['total']:,} total │ {params['trainable']:,} trainable │ {params['total'] - params['trainable']:,} frozen")
    log.info(f"    Ratio     : audio:text = {params['audio_text_ratio']}")
    log.info("")
    log.info("  DATA")
    log.info(f"    Train     : {len(train_data)} participants")
    log.info(f"    Dev       : {len(dev_data)} participants")
    log.info(f"    Test      : {len(test_data)} participants")
    log.info(f"    Features  : text(384) + mfcc(120) + egemaps(88), aligned per segment")
    log.info(f"    PHQ range : [{train_labels.min():.0f}, {train_labels.max():.0f}]  mean={train_labels.mean():.1f}")
    log.info("")
    log.info("  TRAINING")
    log.info(f"    Optimizer : AdamW (lr={args.lr}, weight_decay={args.wd})")
    log.info(f"    Scheduler : ReduceOnPlateau (factor=0.5, patience={args.sched_patience})")
    log.info(f"    Loss      : {'CCC (1 - CCC)' if args.loss == 'ccc' else f'WeightedMSE (high_weight={args.high_weight} for PHQ≥{args.phq_thresh})'}")
    log.info(f"    Epochs    : {args.epochs} (early stopping patience={args.patience})")
    log.info(sep)

    # ── Optionally unfreeze text branch for joint fine-tuning ──
    if args.unfreeze_text:
        model.unfreeze_text()
        params = model.param_summary()  # re-compute after unfreeze
        log.info(f"  TEXT BRANCH UNFROZEN (text_lr={args.text_lr})")

    # ── Optimizer / Scheduler / Loss ──
    if args.unfreeze_text:
        text_params = list(model.text_bn.parameters()) + list(model.text_head.parameters())
        audio_params = list(model.residual_bn.parameters()) + list(model.residual_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": text_params, "lr": args.text_lr},
            {"params": audio_params, "lr": args.lr},
        ], weight_decay=args.wd)
        trainable_params = text_params + audio_params
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.sched_patience, min_lr=1e-6
    )
    if args.loss == "ccc":
        criterion = CCCLoss()
    elif args.loss == "combined":
        criterion = CombinedLoss(
            phq_threshold=args.phq_thresh, high_weight=args.high_weight,
            ccc_weight=0.5, n_bins=5, floor_weight=0.5, ceil_weight=5.0,
        )
        criterion.fit(train_labels)
        criterion.to(device)
    else:
        criterion = WeightedMSELoss(
            phq_threshold=args.phq_thresh, high_weight=args.high_weight, low_weight=1.0
        )

    train_loader = DataLoader(
        train_data, batch_size=len(train_data), shuffle=True, collate_fn=collate_multimodal
    )

    # ── Evaluate text-only baseline ──
    _, _, baseline = evaluate(model, dev_data, device)
    log.info(f"  Baseline (text-only) │ Val CCC: {baseline['ccc']:.4f} │ Val RMSE: {baseline['rmse']:.3f}")
    log.info(sep)

    # ── Training Header ──
    log.info(f"  {'Epoch':>7} │ {'Loss':>8} │ {'Val CCC':>9} │ {'Val RMSE':>9} │ {'Residual':>9} │ {'LR':>10} │ Status")
    log.info(f"  {'─'*7} │ {'─'*8} │ {'─'*9} │ {'─'*9} │ {'─'*9} │ {'─'*10} │ {'─'*10}")

    best_ccc = -1.0
    best_epoch = 0
    patience_counter = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for text, mfcc_b, egemaps_b, behavioral_b, labels, mask in train_loader:
            text = text.to(device)
            mfcc_b = mfcc_b.to(device)
            egemaps_b = egemaps_b.to(device)
            behavioral_b = behavioral_b.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            preds = model(text, mfcc_b, egemaps_b, mask, behavioral_b).squeeze(-1)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # ── Validation ──
        _, _, val_metrics = evaluate(model, dev_data, device)
        val_ccc = val_metrics["ccc"]
        val_rmse = val_metrics["rmse"]
        residual_mag = float(model.residual_head.weight.data.norm().item())
        lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_ccc)

        # ── Check improvement ──
        status = ""
        if val_ccc > best_ccc + 1e-4:
            best_ccc = val_ccc
            best_epoch = epoch
            patience_counter = 0
            status = "★ best"

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ccc": val_ccc,
                "val_rmse": val_rmse,
                "residual_mag": residual_mag,
                "params": params,
            }, save_dir / "best_fusion.pt")
        else:
            patience_counter += 1

        # ── Log (every epoch to file, key epochs to console) ──
        msg = (
            f"  {epoch:>7} │ {epoch_loss:>8.2f} │ {val_ccc:>9.4f} │ {val_rmse:>9.3f} │ "
            f"{residual_mag:>9.5f} │ {lr:>10.1e} │ {status}"
        )

        if status or epoch <= 5 or epoch % 50 == 0 or epoch == args.epochs:
            log.info(msg)
        else:
            log.debug(msg)

        # ── Early stopping ──
        if patience_counter >= args.patience:
            log.info(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - t_start

    # ── Training Summary ──
    log.info(sep)
    log.info("  TRAINING COMPLETE")
    log.info(f"    Best Val CCC   : {best_ccc:.4f} (epoch {best_epoch})")
    log.info(f"    Residual ‖W‖   : {residual_mag:.5f}")
    log.info(f"    Time           : {elapsed/60:.1f} min ({elapsed/epoch:.2f}s/epoch)")
    log.info(sep)

    # ── Test Evaluation ──
    ckpt = torch.load(save_dir / "best_fusion.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, _, test_metrics = evaluate(model, test_data, device)
    log.info("  TEST EVALUATION")
    log.info(f"    CCC  : {test_metrics['ccc']:.4f}")
    log.info(f"    RMSE : {test_metrics['rmse']:.3f}")
    log.info(f"    MAE  : {test_metrics['mae']:.3f}")
    log.info(sep)

    return best_ccc, test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train Multimodal Fusion Model")
    p.add_argument("--train-csv", default="data/splits/train_split.csv")
    p.add_argument("--dev-csv", default="data/splits/dev_split.csv")
    p.add_argument("--test-csv", default="data/splits/test_split.csv")
    p.add_argument("--feature-dir", default="data/features")
    p.add_argument("--data-dir", default="data/raw",
                   help="Directory with raw participant data (transcripts)")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                   help="Pretrained text-only model checkpoint")
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--sched-patience", type=int, default=20)
    p.add_argument("--phq-thresh", type=float, default=10.0)
    p.add_argument("--high-weight", type=float, default=2.0)
    p.add_argument("--loss", choices=["mse", "ccc", "combined"], default="mse",
                   help="Loss function: weighted MSE, CCC loss, or combined (CWMSE+CCC)")
    p.add_argument("--unfreeze-text", action="store_true",
                   help="Unfreeze text branch with --text-lr (joint fine-tuning)")
    p.add_argument("--text-lr", type=float, default=0.0001,
                   help="Learning rate for text branch when unfrozen")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout on residual branch (before the linear head)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    log_path = setup_logger("logs", prefix="fusion_training")
    logging.getLogger("train").info(f"Log file: {log_path}")

    train(args)


if __name__ == "__main__":
    main()
