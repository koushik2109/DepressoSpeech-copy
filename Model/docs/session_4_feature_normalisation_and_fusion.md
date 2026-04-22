# Session 4: Feature Normalization & Fusion

## Purpose

Different feature modalities have vastly different scales and distributions.
Normalization ensures no single modality dominates, and fusion combines them
into a single feature vector for downstream processing.

---

## The Scale Problem

Without normalization:

| Modality | Typical Range | Magnitude |
|----------|--------------|-----------|
| eGeMAPS (F0 mean) | 80–300 Hz | ~10² |
| eGeMAPS (jitter) | 0.001–0.05 | ~10⁻² |
| MFCC coefficient 1 | -500 to +500 | ~10² |
| MFCC coefficient 40 | -2 to +2 | ~10⁰ |
| SBERT dimension | -0.3 to +0.3 | ~10⁻¹ |

If we concatenate raw features, PCA and the model would be dominated by
high-magnitude features (F0, MFCC-1), ignoring subtle but critical features
like jitter and shimmer.

---

## Normalization Strategy

**File**: `src/features/normalizer.py`

| Modality | Method | Formula | Reason |
|----------|--------|---------|--------|
| eGeMAPS (88) | StandardScaler | $z = \frac{x - \mu}{\sigma}$ | Gaussian-like distributions; unit variance |
| MFCC (120) | StandardScaler | $z = \frac{x - \mu}{\sigma}$ | Same rationale; derived from spectral analysis |
| Text (384) | L2 Normalization | $\hat{x} = \frac{x}{\|x\|_2}$ | SBERT embeddings are directional; L2 preserves angular relationships |

### Why Different Methods?

**StandardScaler for acoustic features**: eGeMAPS and MFCC features follow
roughly Gaussian distributions. StandardScaler centers at 0 with unit variance,
making all features comparable.

**L2 for text embeddings**: SBERT embeddings encode meaning in their direction,
not magnitude. L2 normalization projects all embeddings onto the unit sphere,
preserving cosine similarity (the metric SBERT was trained to optimize).

### Critical Rule: Fit on Training Data Only

```python
normalizer = FeatureNormalizer()

# Fit ONLY on training split
normalizer.fit(train_egemaps, train_mfcc)

# Transform ALL splits using training statistics
train_normed = normalizer.transform(train_egemaps, train_mfcc, train_text)
dev_normed   = normalizer.transform(dev_egemaps, dev_mfcc, dev_text)
test_normed  = normalizer.transform(test_egemaps, test_mfcc, test_text)

# Save for inference parity
normalizer.save()  # → checkpoints/scalers/feature_scalers.pkl
```

**Why?** Using dev/test statistics would leak future information into preprocessing,
inflating evaluation metrics. This is a common mistake in ML pipelines.

### Zero-Vector Protection

If SBERT produces a zero vector (e.g., failed transcription), L2 normalization
would divide by zero. The normalizer handles this:

```python
norms = np.linalg.norm(text_features, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)  # prevent division by zero
normalized = text_features / norms
```

---

## Fusion

**File**: `src/features/fusion.py`

Simple concatenation along the feature axis:

$$\mathbf{f}_{\text{fused}} = [\mathbf{f}_{\text{egemaps}} \| \mathbf{f}_{\text{mfcc}} \| \mathbf{f}_{\text{text}}]$$

$$\text{Shape: } (N, 88) \| (N, 120) \| (N, 384) \rightarrow (N, 592)$$

```python
fusion = FeatureFusion()
fused = fusion.fuse({
    "egemaps": normed_egemaps,  # (N, 88)
    "mfcc": normed_mfcc,        # (N, 120)
    "text": normed_text          # (N, 384)
})
# fused.shape = (N, 592)
```

### Why Simple Concatenation?

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| Concatenation | Zero parameters, preserves all info | No cross-modal learning | ✅ Best for small dataset |
| Learned attention fusion | Weights modalities adaptively | Adds parameters; needs data | ❌ Overfitting risk |
| Cross-modal transformer | Rich interactions | 1000s of parameters | ❌ Way too large |

With 163 training participants, any learned fusion adds overfitting risk.
Let PCA and the model learn cross-modal relationships instead.

---

## Session 4.5: PCA Dimensionality Reduction

**File**: `src/features/pca_reducer.py`

### The Problem

592 features × ~1,700 model parameters × 163 training samples.

The model is too small to process 592-dim input directly:
- Linear(592→16) alone = 9,488 parameters (5.6× our entire budget!)
- Would need ~95,000 training samples for a healthy ratio

### The Solution

$$\text{PCA: } \mathbb{R}^{592} \rightarrow \mathbb{R}^{64}$$

- Reduce 592 → 64 dimensions using PCA
- **Linear(64→16) = 1,040 parameters** (fits our budget)
- Retains ~93% of variance
- Acts as implicit regularizer (removes noise dimensions)

### PCA Stability Check

$$\frac{N_{\text{timesteps}}}{N_{\text{features}}} = \frac{16{,}300}{592} \approx 27.5 \gg 5$$

PCA is fit on all **timesteps** across all training participants (not per-participant),
giving ~16,300 samples for 592 features. The covariance estimate is stable.

### Variance Retention by Modality

| Feature Group | Raw Dims | Typical PCs Needed for 95% | Notes |
|---------------|----------|---------------------------|-------|
| eGeMAPS | 88 | ~10–12 | Correlated features compress well |
| MFCC+Δ+ΔΔ | 120 | ~15–20 | Deltas are derived from base MFCCs |
| SBERT | 384 | ~30–35 | Dense embeddings; some information loss |
| **Total** | **592** | **~55–67** | **64 components is well-placed** |

### Usage

```python
pca = PCAReducer(n_components=64)

# Fit ONLY on training data
pca.fit(train_fused_592)

# Transform all splits
train_64 = pca.transform(train_fused_592)  # (N_train, 64)
dev_64   = pca.transform(dev_fused_592)    # (N_dev, 64)
test_64  = pca.transform(test_fused_592)   # (N_test, 64)

# Save for inference
pca.save()  # → checkpoints/scalers/pca_reducer.pkl
```

### Input Validation

PCA enforces that input must be exactly 592-dim at both `fit()` and `transform()`.
If a different dimension is passed, it raises a clear error:

```
ValueError: Expected input dim 592, got 384.
Make sure features are fused [eGeMAPS(88) + MFCC(120) + Text(384)] = 592
```

---

## Complete Session 4 Data Flow

```
Training:
  egemaps_train (N, 88)  ─┐
  mfcc_train    (N, 120) ─┤→ normalizer.fit(train)
  text_train    (N, 384) ─┘
                              │
  egemaps_all   (M, 88)  ─┐  │
  mfcc_all      (M, 120) ─┤→ normalizer.transform(all)
  text_all      (M, 384) ─┘
                              │
                         fuse → (M, 592)
                              │
                    PCA.fit(train_592)
                              │
                    PCA.transform(all) → (M, 64)

Inference:
  normalize.load() → transform → fuse → PCA.load() → transform
  Same sequence, no fitting
```

---

## Saved Artifacts

| File | Contents | Created By |
|------|----------|-----------|
| `checkpoints/scalers/feature_scalers.pkl` | StandardScaler for egemaps + mfcc | `FeatureNormalizer.save()` |
| `checkpoints/scalers/pca_reducer.pkl` | Fitted PCA + sklearn version | `PCAReducer.save()` |

**Both artifacts are required for inference.** Missing either file means the
inference pipeline cannot start.

---

## Configuration

**File**: `configs/normalization_config.yaml`

```yaml
egemaps:
  method: "standard_scaler"
mfcc:
  method: "standard_scaler"
text:
  method: "l2_normalize"
scaler:
  save_dir: "checkpoints/scalers"
  filename: "feature_scalers.pkl"
```

---

## Files

| File | Purpose |
|------|---------|
| `src/features/normalizer.py` | StandardScaler + L2 normalization |
| `src/features/fusion.py` | Concatenation fusion |
| `src/features/pca_reducer.py` | PCA 592→64 with input validation |
| `configs/normalization_config.yaml` | Scaler settings |
| `configs/model_config.yaml` | PCA section (n_components, input_dim) |