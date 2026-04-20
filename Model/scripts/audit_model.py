#!/usr/bin/env python3
"""
Comprehensive audit of the fusion model:
  1. Training/Validation/Test metrics
  2. Bias check per severity band
  3. Prediction distribution analysis
  4. Data leakage check (split overlap)
  5. Feature quality check (NaN/Inf, alignment)
  6. Audio contribution analysis

Outputs: JSON report + console summary.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multimodal_fusion import MultimodalFusion
from src.training.metrics import concordance_correlation_coefficient as ccc_metric

# ── Reuse data loading from train_fusion ──
from scripts.train_fusion import load_split, collate_multimodal, extract_behavioral
from torch.utils.data import DataLoader


def severity_band(phq):
    if phq < 5: return "none/minimal (0-4)"
    if phq < 10: return "mild (5-9)"
    if phq < 15: return "moderate (10-14)"
    if phq < 20: return "mod-severe (15-19)"
    return "severe (20-24)"


def evaluate_split(model, data, device):
    """Run inference, return per-participant predictions and targets."""
    model.eval()
    loader = DataLoader(data, batch_size=len(data), shuffle=False, collate_fn=collate_multimodal)
    pids = [d["pid"] for d in data]
    all_preds, all_targets = [], []

    with torch.no_grad():
        for text, mfcc, egemaps, behavioral, labels, mask in loader:
            preds = model(
                text.to(device), mfcc.to(device), egemaps.to(device),
                mask.to(device), behavioral.to(device)
            ).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return pids, preds, targets


def evaluate_text_only(model, data, device):
    """Evaluate text branch only (zero out residual)."""
    model.eval()
    # Save original weights
    orig_w = model.residual_head.weight.data.clone()
    orig_b = model.residual_head.bias.data.clone()
    # Zero out residual
    model.residual_head.weight.data.zero_()
    model.residual_head.bias.data.zero_()

    _, preds, targets = evaluate_split(model, data, device)

    # Restore
    model.residual_head.weight.data.copy_(orig_w)
    model.residual_head.bias.data.copy_(orig_b)

    return preds, targets


def per_band_metrics(preds, targets):
    """Compute MAE and count per severity band."""
    bands = defaultdict(lambda: {"preds": [], "targets": []})
    for p, t in zip(preds, targets):
        band = severity_band(t)
        bands[band]["preds"].append(p)
        bands[band]["targets"].append(t)

    results = {}
    for band in sorted(bands.keys()):
        d = bands[band]
        p = np.array(d["preds"])
        t = np.array(d["targets"])
        results[band] = {
            "n": len(t),
            "true_mean": round(float(t.mean()), 2),
            "pred_mean": round(float(p.mean()), 2),
            "mae": round(float(np.abs(p - t).mean()), 2),
            "mean_error": round(float((p - t).mean()), 2),  # +ve = over-predict
            "rmse": round(float(np.sqrt(np.mean((p - t) ** 2))), 2),
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"timestamp": datetime.now().isoformat(), "device": str(device)}

    # ── 1. Data leakage check ──
    print("=" * 70)
    print("  AUDIT 1: Data Leakage Check")
    print("=" * 70)
    train_csv = pd.read_csv("data/splits/train_split.csv")
    dev_csv = pd.read_csv("data/splits/dev_split.csv")
    test_csv = pd.read_csv("data/splits/test_split.csv")

    train_ids = set(train_csv["Participant_ID"].astype(str))
    dev_ids = set(dev_csv["Participant_ID"].astype(str))
    test_ids = set(test_csv["Participant_ID"].astype(str))

    leaks = {
        "train_dev_overlap": sorted(train_ids & dev_ids),
        "train_test_overlap": sorted(train_ids & test_ids),
        "dev_test_overlap": sorted(dev_ids & test_ids),
    }
    report["data_leakage"] = leaks
    any_leak = any(len(v) > 0 for v in leaks.values())
    print(f"  Train-Dev overlap  : {len(leaks['train_dev_overlap'])} {'⚠️ LEAK!' if leaks['train_dev_overlap'] else '✅ clean'}")
    print(f"  Train-Test overlap : {len(leaks['train_test_overlap'])} {'⚠️ LEAK!' if leaks['train_test_overlap'] else '✅ clean'}")
    print(f"  Dev-Test overlap   : {len(leaks['dev_test_overlap'])} {'⚠️ LEAK!' if leaks['dev_test_overlap'] else '✅ clean'}")

    # ── 2. Label distribution ──
    print("\n" + "=" * 70)
    print("  AUDIT 2: Label Distribution")
    print("=" * 70)
    label_dist = {}
    band_ranges = [
        ("none/minimal (0-4)", 0, 4),
        ("mild (5-9)", 5, 9),
        ("moderate (10-14)", 10, 14),
        ("mod-severe (15-19)", 15, 19),
        ("severe (20-24)", 20, 24),
    ]
    for name, df in [("train", train_csv), ("dev", dev_csv), ("test", test_csv)]:
        scores = df["PHQ_Score"].values
        dist = {}
        for band_name, low, high in band_ranges:
            count = int(((scores >= low) & (scores <= high)).sum())
            pct = round(count / len(scores) * 100, 1)
            dist[band_name] = {"n": count, "pct": pct}
        label_dist[name] = {
            "total": len(scores),
            "mean": round(float(scores.mean()), 2),
            "std": round(float(scores.std()), 2),
            "min": int(scores.min()),
            "max": int(scores.max()),
            "bands": dist,
        }
        print(f"\n  {name.upper()} (n={len(scores)}, mean={scores.mean():.1f}, std={scores.std():.1f})")
        for b, v in dist.items():
            bar = "█" * (v["n"] * 2)
            print(f"    {b:25s} : {v['n']:3d} ({v['pct']:5.1f}%) {bar}")

    report["label_distribution"] = label_dist

    # ── 3. Feature quality check ──
    print("\n" + "=" * 70)
    print("  AUDIT 3: Feature Quality Check")
    print("=" * 70)
    feature_dir = Path("data/features")
    all_pids = train_ids | dev_ids | test_ids
    feature_issues = {"missing": [], "nan_inf": [], "shape_mismatch": [], "behavioral_missing": []}
    for pid in sorted(all_pids):
        npz_path = feature_dir / f"{pid}_training.npz"
        if not npz_path.exists():
            feature_issues["missing"].append(pid)
            continue
        f = np.load(npz_path)
        text = f["text_embeddings"]
        mfcc = f["mfcc"]
        egemaps = f["egemaps"]
        # NaN/Inf
        for name, arr in [("text", text), ("mfcc", mfcc), ("egemaps", egemaps)]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                feature_issues["nan_inf"].append(f"{pid}:{name}")
        # Shape alignment
        if not (text.shape[0] == mfcc.shape[0] == egemaps.shape[0]):
            feature_issues["shape_mismatch"].append(pid)
        # Behavioral
        beh = extract_behavioral(pid, "data/raw")
        if beh is None:
            feature_issues["behavioral_missing"].append(pid)

    report["feature_quality"] = {k: v for k, v in feature_issues.items()}
    print(f"  Missing NPZ files     : {len(feature_issues['missing'])} {'⚠️' if feature_issues['missing'] else '✅'}")
    print(f"  NaN/Inf values        : {len(feature_issues['nan_inf'])} {'⚠️' if feature_issues['nan_inf'] else '✅'}")
    print(f"  Shape mismatches      : {len(feature_issues['shape_mismatch'])} {'⚠️' if feature_issues['shape_mismatch'] else '✅'}")
    print(f"  Behavioral missing    : {len(feature_issues['behavioral_missing'])} {'⚠️' if feature_issues['behavioral_missing'] else '✅'}")
    if feature_issues["behavioral_missing"]:
        print(f"    PIDs without transcripts: {feature_issues['behavioral_missing'][:10]}...")

    # ── 4. Load model & run predictions ──
    print("\n" + "=" * 70)
    print("  AUDIT 4: Model Performance & Bias Analysis")
    print("=" * 70)
    model = MultimodalFusion(stats_mode="mean_std").to(device)
    model.load_pretrained_text("checkpoints/best_model.pt")

    ckpt = torch.load("checkpoints/best_fusion.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded fusion checkpoint (epoch {ckpt['epoch']}, val_ccc={ckpt['val_ccc']:.4f})")

    feature_dir = Path("data/features")
    train_data = load_split(Path("data/splits/train_split.csv"), feature_dir, "data/raw")
    dev_data = load_split(Path("data/splits/dev_split.csv"), feature_dir, "data/raw")
    test_data = load_split(Path("data/splits/test_split.csv"), feature_dir, "data/raw")

    perf = {}
    for split_name, split_data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        pids, preds, targets = evaluate_split(model, split_data, device)

        ccc = ccc_metric(preds, targets)
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        mae = float(np.mean(np.abs(preds - targets)))

        # Per-band analysis
        band_metrics = per_band_metrics(preds, targets)

        # Prediction distribution
        pred_min = float(preds.min())
        pred_max = float(preds.max())
        pred_mean = float(preds.mean())
        pred_std = float(preds.std())
        true_std = float(targets.std())

        perf[split_name] = {
            "n": len(preds),
            "ccc": round(ccc, 4),
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "pred_range": [round(pred_min, 2), round(pred_max, 2)],
            "pred_mean": round(pred_mean, 2),
            "pred_std": round(pred_std, 2),
            "true_mean": round(float(targets.mean()), 2),
            "true_std": round(true_std, 2),
            "std_ratio": round(pred_std / max(true_std, 1e-8), 3),
            "per_band": band_metrics,
        }

        print(f"\n  {split_name.upper()} (n={len(preds)})")
        print(f"    CCC : {ccc:.4f}  |  RMSE : {rmse:.3f}  |  MAE : {mae:.3f}")
        print(f"    Pred range: [{pred_min:.1f}, {pred_max:.1f}]  mean={pred_mean:.1f}  std={pred_std:.1f}")
        print(f"    True range: [{targets.min():.1f}, {targets.max():.1f}]  mean={targets.mean():.1f}  std={true_std:.1f}")
        print(f"    Std ratio (pred/true): {pred_std/max(true_std,1e-8):.3f} {'⚠️ collapsed!' if pred_std/max(true_std,1e-8) < 0.5 else '✅'}")
        print(f"\n    Per-severity-band breakdown:")
        print(f"    {'Band':28s} {'N':>4s} {'True':>7s} {'Pred':>7s} {'MAE':>7s} {'Bias':>7s}")
        print(f"    {'─'*28} {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for band, bm in band_metrics.items():
            bias_str = f"{bm['mean_error']:+.1f}"
            flag = " ⚠️" if abs(bm['mean_error']) > 3.0 else ""
            print(f"    {band:28s} {bm['n']:4d} {bm['true_mean']:7.1f} {bm['pred_mean']:7.1f} {bm['mae']:7.1f} {bias_str:>7s}{flag}")

    report["performance"] = perf

    # ── 5. Audio contribution (text-only vs fusion) ──
    print("\n" + "=" * 70)
    print("  AUDIT 5: Audio + Behavioral Contribution")
    print("=" * 70)

    contrib = {}
    for split_name, split_data in [("dev", dev_data), ("test", test_data)]:
        text_preds, targets = evaluate_text_only(model, split_data, device)
        _, fusion_preds, _ = evaluate_split(model, split_data, device)
        # No clamping — raw predictions match training evaluation

        text_ccc = ccc_metric(text_preds, targets)
        fusion_ccc = ccc_metric(fusion_preds, targets)
        residual = fusion_preds - text_preds

        contrib[split_name] = {
            "text_only_ccc": round(text_ccc, 4),
            "fusion_ccc": round(fusion_ccc, 4),
            "improvement": round(fusion_ccc - text_ccc, 4),
            "residual_mean": round(float(residual.mean()), 3),
            "residual_std": round(float(residual.std()), 3),
            "residual_range": [round(float(residual.min()), 3), round(float(residual.max()), 3)],
        }

        print(f"\n  {split_name.upper()}")
        print(f"    Text-only CCC : {text_ccc:.4f}")
        print(f"    Fusion CCC    : {fusion_ccc:.4f}")
        print(f"    Improvement   : {fusion_ccc - text_ccc:+.4f}")
        print(f"    Residual mean : {residual.mean():.3f}  std : {residual.std():.3f}")
        print(f"    Residual range: [{residual.min():.2f}, {residual.max():.2f}]")

    report["audio_contribution"] = contrib

    # ── 6. Residual weight analysis ──
    print("\n" + "=" * 70)
    print("  AUDIT 6: Model Weight Analysis")
    print("=" * 70)
    w = model.residual_head.weight.data.cpu().numpy().flatten()
    audio_w = w[:416]  # first 416 = audio stats-pooled
    behav_w = w[416:]  # last 16 = behavioral

    weight_analysis = {
        "residual_weight_norm": round(float(np.linalg.norm(w)), 5),
        "audio_weight_norm": round(float(np.linalg.norm(audio_w)), 5),
        "behavioral_weight_norm": round(float(np.linalg.norm(behav_w)), 5),
        "audio_weight_mean_abs": round(float(np.abs(audio_w).mean()), 6),
        "behavioral_weight_mean_abs": round(float(np.abs(behav_w).mean()), 6),
        "top5_behavioral_weights": [],
    }

    behav_names = [
        "n_turns", "avg_duration", "duration_std", "avg_pause", "pause_std",
        "median_pause", "long_pause_frac", "avg_words", "words_std",
        "speaking_rate", "rate_std", "total_speaking_time", "speaking_ratio",
        "interview_duration", "max_turn_duration", "total_word_count"
    ]
    behav_importance = sorted(
        zip(behav_names, behav_w.tolist()),
        key=lambda x: abs(x[1]), reverse=True
    )
    weight_analysis["top5_behavioral_weights"] = [
        {"feature": name, "weight": round(w_val, 6)} for name, w_val in behav_importance[:5]
    ]

    report["weight_analysis"] = weight_analysis
    print(f"  Total residual ‖W‖: {np.linalg.norm(w):.5f}")
    print(f"  Audio ‖W‖  : {np.linalg.norm(audio_w):.5f}  ({np.linalg.norm(audio_w)/np.linalg.norm(w)*100:.1f}%)")
    print(f"  Behav ‖W‖  : {np.linalg.norm(behav_w):.5f}  ({np.linalg.norm(behav_w)/np.linalg.norm(w)*100:.1f}%)")
    print(f"\n  Top behavioral feature weights:")
    for name, w_val in behav_importance[:8]:
        print(f"    {name:25s} : {w_val:+.6f}")

    # ── 7. Summary ──
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    issues = []
    # Check bias
    for split_name in ["dev", "test"]:
        for band, bm in perf[split_name]["per_band"].items():
            if abs(bm["mean_error"]) > 3.0:
                issues.append(f"{split_name}/{band}: bias={bm['mean_error']:+.1f}")
    # Check std collapse
    for split_name in ["dev", "test"]:
        sr = perf[split_name]["std_ratio"]
        if sr < 0.5:
            issues.append(f"{split_name}: prediction std collapsed (ratio={sr:.2f})")
    # Check data leakage
    if any_leak:
        issues.append("DATA LEAKAGE DETECTED")

    report["issues"] = issues
    report["summary"] = {
        "val_ccc": perf["dev"]["ccc"],
        "test_ccc": perf["test"]["ccc"],
        "train_ccc": perf["train"]["ccc"],
        "audio_improvement_val": contrib["dev"]["improvement"],
        "audio_improvement_test": contrib["test"]["improvement"],
        "total_issues": len(issues),
        "ready_for_inference": len(issues) == 0 and perf["dev"]["ccc"] >= 0.5,
    }

    if issues:
        print(f"\n  ⚠️  ISSUES FOUND ({len(issues)}):")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"\n  ✅ No critical issues found!")

    print(f"\n  Val CCC  : {perf['dev']['ccc']:.4f}")
    print(f"  Test CCC : {perf['test']['ccc']:.4f}")
    print(f"  Audio+Behav contribution: val {contrib['dev']['improvement']:+.4f}, test {contrib['test']['improvement']:+.4f}")

    # Save JSON report
    report_path = Path("logs/audit_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
