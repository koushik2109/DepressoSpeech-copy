# DepressoSpeech — Session Plan

## Medical-Grade Depression Detection from Speech

> **Goal**: Predict PHQ-8 depression severity scores (0–24) from speech audio
> using the DAIC-WOZ clinical interview dataset.

---

## Architecture Overview

```
Audio (.wav) → Preprocessing → Feature Extraction → Normalization → Fusion → PCA → Model → PHQ-8 Score
                                                                                          ↕
                                                                              Training ← Loss + Metrics
                                                                              Inference → API Response
```

### Model: MLP → BiGRU → Attention → Linear

```
Input (B, T, 64)
  → MLP: Linear(64→16) + LayerNorm + ReLU + Dropout(0.5)        [1,072 params]
  → BiGRU: GRU(16→4, bidirectional) + LayerNorm + Dropout(0.5)  [504 params]
  → Attention: Additive attention pooling over time              [80 params]
  → Head: Dropout(0.5) + Linear(8→1)                            [9 params]
  → Output: PHQ-8 score (scalar)                    TOTAL: ~1,700 params
```

> **Why GRU over LSTM?** GRU has 3 gates vs LSTM's 4 gates = 33% fewer parameters.
> With only 163 training participants, every parameter matters.

---

## Session Breakdown

| Session | Topic | Status | Key Deliverable |
|---------|-------|--------|-----------------|
| 1 | Project Setup & Data Exploration | ⬜ Pending | Repository structure, dependencies, EDA notebook |
| 2 | Audio Preprocessing Pipeline | ✅ Complete | VAD + chunking (5s, 25% overlap, 16kHz) |
| 3 | Feature Extraction (3 modalities) | ✅ Complete | eGeMAPS(88) + MFCC(120) + SBERT(384) |
| 4 | Feature Normalization & Fusion | ✅ Complete | StandardScaler + L2 + concat → 592-dim |
| 4.5 | PCA Dimensionality Reduction | ✅ Complete | 592 → 64 dims (~93% variance retained) |
| 5 | Dataset & DataLoader | ✅ Complete | Padded sequences + masks + sorted batching |
| 6 | Model Architecture | ✅ Complete | MLP→BiGRU→Attention→Linear (~1,700 params) |
| 7 | Training Pipeline | ✅ Complete | Trainer + CCC metric + early stopping + checkpoints |
| 8 | Inference Pipeline | ✅ Complete | Audio→score chain + CLI predictor |
| 9 | REST API (FastAPI) | ✅ Complete | /predict, /predict/batch, /health endpoints |
| 10 | Database & Logging | ✅ Complete | SQLite tracking + centralized file logging |

---

## Data: DAIC-WOZ Dataset

| Split | Participants | PHQ-8 Range | Purpose |
|-------|-------------|-------------|---------|
| Train | 163 | 0–24 | Fit scalers, PCA, model weights |
| Dev | 56 | 0–24 | Early stopping, hyperparameter selection |
| Test | 56 | 0–24 | Final evaluation (untouched until end) |

**Label distribution**: ~86% PHQ < 10 (non-depressed), ~14% PHQ ≥ 10 (depressed)
→ Addressed with WeightedMSE loss (2× weight for PHQ ≥ 10)

---

## Dimension Flow (End-to-End)

```
                    TRAINING                          INFERENCE
                    ────────                          ─────────
              CSV feature files                    Raw audio (.wav)
                     │                                    │
         ┌───────────┼───────────┐          ┌─────────────┼──────────────┐
         │           │           │          │             │              │
    eGeMAPS(88)  MFCC(120)  SBERT(384)  eGeMAPS(88)  MFCC(120)   SBERT(384)
         │           │           │          │             │              │
         └───────────┼───────────┘          └─────────────┼──────────────┘
                     │                                    │
              Normalize (fit)                      Normalize (load)
                     │                                    │
              Fuse → (592)                         Fuse → (592)
                     │                                    │
              PCA fit → (64)                       PCA load → (64)
                     │                                    │
              DataLoader                           DataLoader
           (B, T_max, 64)                         (1, C, 64)
                     │                                    │
                Model.train()                      Model.eval()
           MLP→BiGRU→Attn→Head                MLP→BiGRU→Attn→Head
                     │                                    │
              Loss + Backward                     PHQ-8 score [0,24]
```

---

## Running the Pipeline

### Training
```bash
# 1. Extract features from DAIC-WOZ CSVs
python scripts/extract_features.py --config configs/feature_config.yaml

# 2. Train model
python scripts/train.py --config configs/training_config.yaml

# Output: checkpoints/best_model.pt, checkpoints/scalers/*.pkl
```

### Inference
```bash
# Single file prediction
python scripts/predict.py --audio path/to/audio.wav

# Batch prediction
python scripts/predict.py --audio-dir path/to/audio_folder/ --output results.csv
```

### API Server
```bash
# Start FastAPI server
python scripts/serve.py --config configs/inference_config.yaml --port 8000

# Endpoints:
#   POST /predict       — single audio file → PHQ-8 score
#   POST /predict/batch — multiple files → batch results
#   GET  /health        — model status check
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| GRU over LSTM | 33% fewer params; comparable on short sequences |
| PCA 592→64 | Implicit regularizer; stable with 16K+ timesteps |
| Dropout 0.5 everywhere | Aggressive regularization for 163 samples |
| CCC as primary metric | Standard for AVEC/DAIC-WOZ depression prediction |
| WeightedMSE loss | 2× weight for PHQ≥10 to address class imbalance |
| Early stopping patience=5 | Prevents overfitting on small dataset |
| ~1,700 params total | 10.3:1 ratio with 163 training participants |

---

## Expected Performance

| Metric | Expected Range | Published Baselines |
|--------|---------------|-------------------|
| CCC | 0.35 – 0.55 | 0.4 – 0.7 (large models) |
| RMSE | 5.0 – 7.0 | 4.0 – 6.0 (large models) |
| MAE | 3.5 – 5.5 | 3.0 – 5.0 (large models) |

> Our model is intentionally small (~1,700 params vs 10M+ in SOTA).
> Lower performance is expected but acceptable for a screening tool.