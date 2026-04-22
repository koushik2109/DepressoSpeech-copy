# Session 8: Inference Pipeline

## Purpose

Take a raw audio file and produce a PHQ-8 depression severity prediction.
The inference pipeline must exactly mirror the training preprocessing to
ensure feature parity.

---

## Pipeline Flow

```
audio.wav
    │
    ▼
[AudioPreprocessor]  resample → VAD → chunk
    │ (C, 80000) numpy arrays
    │
    ├──→ [EgemapsExtractor.extract_from_audio()]  → (C, 88)
    ├──→ [MfccExtractor.extract_from_audio()]     → (C, 120)
    └──→ [TextExtractor.extract_from_audio()]     → (C, 384)
    │
    ▼
[FeatureNormalizer.load() → transform()]
    │ (C, 88), (C, 120), (C, 384) — normalized
    │
    ▼
[FeatureFusion.fuse()]  → (C, 592)
    │
    ▼
[PCAReducer.load() → transform()]  → (C, 64)
    │
    ▼
[Predictor]  numpy → tensor → model.eval() → PHQ-8 score
    │
    ▼
PredictionResult
    ├── phq8_score: 12.3  (clamped to [0, 24])
    ├── severity: "moderate"
    └── num_chunks: 15
```

---

## Train-Inference Parity

The most critical requirement: **inference must apply identical transformations
as training.** Any mismatch means the model receives out-of-distribution input.

| Step | Training | Inference | Parity Mechanism |
|------|----------|-----------|-----------------|
| Normalization | `fit()` on train split | `load()` fitted scalers | `feature_scalers.pkl` |
| PCA | `fit()` on train split | `load()` fitted PCA | `pca_reducer.pkl` |
| Feature extraction | Same extractors | Same extractors | Shared code |
| Model weights | `train()` → save | `load()` checkpoint | `best_model.pt` |

**If any of the 3 artifacts are missing, the pipeline refuses to start.**

---

## Components

### 1. InferencePipeline (Orchestrator)

**File**: `src/inference/pipeline.py`

Coordinates the full audio-to-prediction chain. Lazy-loads heavy components
(Whisper, SBERT, OpenSMILE) only on first use.

```python
pipeline = InferencePipeline(config_path="configs/inference_config.yaml")

# Single prediction
result = pipeline.predict_from_audio("patient_audio.wav")
print(result.phq8_score)   # 12.3
print(result.severity)     # "moderate"
print(result.num_chunks)   # 15
print(result.timing)       # {"preprocess": 0.5, "extraction": 2.1, ...}
```

### Per-Stage Timing

Every inference call logs timing for each stage:

```
[INFERENCE] Preprocessing: 0.52s
[INFERENCE] Feature extraction: 2.13s
[INFERENCE] Normalize + Fuse + PCA: 0.03s
[INFERENCE] Model prediction: 0.01s
[INFERENCE] Total: 2.69s
```

### 2. Predictor (Model Runner)

**File**: `src/inference/predictor.py`

Handles model loading and tensor management.

```python
predictor = Predictor(
    checkpoint_path="checkpoints/best_model.pt",
    device="cuda"  # or "cpu"
)
score = predictor.predict(features_64)  # numpy (C, 64) → float
```

**Key behaviors:**
- Accepts both `np.ndarray` and `torch.Tensor` input
- Adds batch dimension if needed: `(C, 64)` → `(1, C, 64)`
- Creates mask and lengths for the model
- Clamps output to `[0, 24]` (valid PHQ-8 range)
- Uses `@torch.no_grad()` for memory efficiency

### Checkpoint Loading

Handles two checkpoint formats:

```python
# Format 1: Trainer checkpoint (dict with metadata)
{
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "epoch": 42,
    "best_metric": 0.45,
    "model_config": {...}
}

# Format 2: Raw state dict
{"mlp.layers.0.weight": tensor, "bigru.gru.weight_ih_l0": tensor, ...}
```

---

## Severity Classification

PHQ-8 scores are mapped to severity levels using standard PHQ-9 cutoffs
(Kroenke et al., 2001). Note: PHQ-8 has no official published cutoffs;
PHQ-9 thresholds are used as proxy, which is common practice in research.

| Score Range | Severity | Clinical Interpretation |
|-------------|----------|----------------------|
| 0 – 4 | Minimal | No significant symptoms |
| 5 – 9 | Mild | Minor depression, monitor |
| 10 – 14 | Moderate | Active treatment recommended |
| 15 – 19 | Moderately Severe | Medication and/or therapy |
| 20 – 24 | Severe | Immediate clinical attention |

⚠️ **This is a screening tool, NOT a diagnostic instrument.**
Clinical diagnosis requires a qualified mental health professional.

---

## Required Artifacts

| Artifact | Path | Created By | Required? |
|----------|------|-----------|-----------|
| Model checkpoint | `checkpoints/best_model.pt` | `scripts/train.py` | ✅ Yes |
| Feature scalers | `checkpoints/scalers/feature_scalers.pkl` | `scripts/extract_features.py` | ✅ Yes |
| PCA reducer | `checkpoints/scalers/pca_reducer.pkl` | `scripts/extract_features.py` | ✅ Yes |

---

## CLI Usage

**File**: `scripts/predict.py`

```bash
# Single file prediction
python scripts/predict.py --audio path/to/audio.wav
# Output:
#   PHQ-8 Score: 12.3
#   Severity: moderate
#   Chunks processed: 15

# Batch prediction (directory of audio files)
python scripts/predict.py --audio-dir path/to/folder/ --output results.csv
# Output: results.csv with columns [filename, phq8_score, severity, num_chunks]

# Custom config
python scripts/predict.py --audio audio.wav --config custom_inference.yaml
```

---

## Configuration

**File**: `configs/inference_config.yaml`

```yaml
model:
  checkpoint_path: "checkpoints/best_model.pt"
  device: "auto"           # auto-detects GPU/CPU

artifacts:
  normalizer_path: "checkpoints/scalers/feature_scalers.pkl"
  pca_path: "checkpoints/scalers/pca_reducer.pkl"

audio:
  sample_rate: 16000
  chunk_duration: 5.0
  chunk_overlap: 0.25
```

---

## Files

| File | Purpose |
|------|---------|
| `src/inference/pipeline.py` | Full audio→prediction orchestrator |
| `src/inference/predictor.py` | Model loading + tensor management |
| `src/inference/__init__.py` | Package exports |
| `scripts/predict.py` | CLI entry point |
| `configs/inference_config.yaml` | Artifact paths + device settings |