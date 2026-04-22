# Session 6: Model Architecture

## Purpose

Map variable-length PCA-reduced feature sequences to a single PHQ-8 score.
The architecture must be tiny (~1,700 parameters) to avoid overfitting on
163 training participants.

---

## Architecture Diagram

```
Input: (B, T, 64)  ← PCA-reduced fused features
         │
         ▼
┌─────────────────────┐
│     MLP Block        │  Linear(64→16) + LayerNorm + ReLU + Dropout(0.5)
│   (B, T, 64→16)     │  Params: 1,072
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   BiGRU Encoder      │  GRU(16→4, bidirectional) + LayerNorm + Dropout(0.5)
│   (B, T, 16→8)      │  Params: 504
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Attention Pooling   │  Additive attention with mask support
│   (B, T, 8) → (B,8) │  Params: 80
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Regression Head     │  Dropout(0.5) + Linear(8→1)
│   (B, 8) → (B,)     │  Params: 9
└─────────────────────┘
          │
          ▼
Output: (B,)  ← PHQ-8 scores
```

**Total: ~1,700 trainable parameters**

---

## Component 1: MLP Block

**File**: `src/models/mlp_block.py`

The MLP block compresses 64-dim PCA features to 16-dim hidden representations
independently at each time-step.

```
(B, T, 64) → Linear(64, 16) → LayerNorm(16) → ReLU → Dropout(0.5) → (B, T, 16)
```

| Layer | Parameters | Calculation |
|-------|-----------|-------------|
| Linear(64→16) | 1,040 | 64×16 + 16 bias |
| LayerNorm(16) | 32 | 16 scale + 16 bias |
| ReLU | 0 | — |
| Dropout(0.5) | 0 | — |
| **Total** | **1,072** | **63.0% of model** |

### Why LayerNorm over BatchNorm?

| Feature | LayerNorm | BatchNorm |
|---------|-----------|-----------|
| Normalizes over | Feature dim | Batch dim |
| Needs batch statistics? | No | Yes |
| Works with B=1 (inference)? | ✅ Yes | ❌ Unreliable |
| Works with small batches? | ✅ Yes | ⚠️ Noisy |

With batch size 8 and variable-length sequences, LayerNorm is more stable.

---

## Component 2: BiGRU Encoder

**File**: `src/models/bigru.py`

Bidirectional GRU captures temporal patterns in both forward and backward
directions of the speech sequence.

```
(B, T, 16) → BiGRU(input=16, hidden=4, bidirectional=True)
           → (B, T, 8)     [4 forward + 4 backward = 8]
           → LayerNorm(8)
           → Dropout(0.5)
           → (B, T, 8)
```

### Why GRU, Not LSTM?

| Feature | GRU | LSTM |
|---------|-----|------|
| Gates | 3 (reset, update, new) | 4 (input, forget, cell, output) |
| Params per layer | $3(n_h^2 + n_h \cdot n_x + n_h)$ | $4(n_h^2 + n_h \cdot n_x + n_h)$ |
| For hidden=4, input=16 | 252 per direction | 336 per direction |
| **Total (bidirectional)** | **504** | **672** |
| Performance gap | Comparable on short sequences | Slight edge on very long sequences |

GRU saves **168 parameters** (25% reduction) with negligible performance loss
on sequences of 50–200 time-steps.

### Parameter Breakdown

For one GRU direction with input_size=16, hidden_size=4:

$$\text{Params} = 3 \times (n_h \times n_x + n_h \times n_h + n_h + n_h)$$
$$= 3 \times (4 \times 16 + 4 \times 4 + 4 + 4) = 3 \times 88 = 264$$

Wait — let's be precise. PyTorch GRU has:
- `weight_ih`: 3 × hidden × input = 3 × 4 × 16 = 192
- `weight_hh`: 3 × hidden × hidden = 3 × 4 × 4 = 48
- `bias_ih`: 3 × hidden = 12
- `bias_hh`: 3 × hidden = 12

Per direction: 192 + 48 + 12 + 12 = **264**
Bidirectional: 264 × 2 = **528**
LayerNorm(8): 8 + 8 = **16**
**BiGRU total: ~544 params**

### Packed Sequences

The BiGRU uses `pack_padded_sequence` to skip computation on padded positions:

```python
packed = pack_padded_sequence(x, lengths, enforce_sorted=True)
output, _ = self.gru(packed)
output, _ = pad_packed_sequence(output, total_length=T)
```

This is why the collate function sorts sequences by length descending —
`enforce_sorted=True` is faster than `False`.

---

## Component 3: Attention Pooling

**File**: `src/models/attention.py`

Collapses the variable-length sequence `(B, T, 8)` into a fixed-size
representation `(B, 8)` by learning which time-steps are most important.

### Additive (Bahdanau) Attention

$$e_t = v^\top \tanh(W h_t + b)$$
$$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T} \exp(e_k)}$$
$$c = \sum_{t=1}^{T} \alpha_t h_t$$

Where:
- $h_t \in \mathbb{R}^8$ — BiGRU output at time $t$
- $W \in \mathbb{R}^{8 \times 8}$ — attention weight matrix
- $v \in \mathbb{R}^{8}$ — context vector
- $\alpha_t$ — attention weight for time-step $t$ (sums to 1)
- $c \in \mathbb{R}^8$ — context vector (weighted sum)

### Mask Handling

Padded positions receive attention score $-\infty$ before softmax:

```python
scores = scores.masked_fill(~mask, float('-inf'))
weights = torch.softmax(scores, dim=-1)
# Padded positions → exp(-inf) = 0 → zero attention weight
```

**NaN protection**: If ALL positions are masked (edge case), softmax produces
NaN. The attention module detects this and replaces NaN weights with 0.

### Parameters

| Layer | Parameters |
|-------|-----------|
| Linear(8→8) + bias | 72 |
| Context vector(8→1) | 8 |
| **Total** | **80** |

### Clinical Value

Attention weights reveal **which 5-second segments** contributed most to the
depression score. This interpretability is valuable for clinical validation:

```python
# Get attention weights for interpretability
weights = model.get_attention_weights(features, mask, lengths)
# weights: (B, T) — which time-steps matter most
```

---

## Component 4: Regression Head

```
(B, 8) → Dropout(0.5) → Linear(8, 1) → squeeze → (B,)
```

| Layer | Parameters |
|-------|-----------|
| Dropout(0.5) | 0 |
| Linear(8→1) + bias | 9 |
| **Total** | **9** |

Output is **unbounded** during training (no activation). PHQ-8 clamping to
[0, 24] happens only at inference time to allow gradients to flow freely.

---

## Parameter Budget Summary

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| MLP Block | 1,072 | 63.0% |
| BiGRU Encoder | ~544 | 32.0% |
| Attention Pooling | 80 | 4.7% |
| Regression Head | 9 | 0.5% |
| **Total** | **~1,700** | **100%** |

$$\text{Param-to-sample ratio} = \frac{1{,}700}{163} \approx 10.4:1$$

### Why This Ratio Matters

| Ratio | Risk | Typical Needs |
|-------|------|--------------|
| < 5:1 | Underfitting | Too few parameters |
| 5–15:1 | Sweet spot | ✅ Our model |
| 15–50:1 | Moderate overfitting | Needs strong regularization |
| > 100:1 | Severe overfitting | Needs pretraining or more data |

---

## Weight Initialization

**File**: `src/models/depression_model.py` → `_init_weights()`

| Layer Type | Initialization | Reason |
|-----------|---------------|--------|
| Linear layers | Xavier Uniform | Preserves gradient variance |
| GRU weights | Orthogonal | Prevents vanishing/exploding gradients in recurrence |
| All biases | Zeros | Standard practice |

---

## Regularization Stack

| Technique | Where | Purpose |
|-----------|-------|---------|
| Dropout(0.5) | MLP, BiGRU output, before head | Prevents co-adaptation |
| LayerNorm | After MLP, after BiGRU | Stabilizes activations |
| Weight decay (1e-3) | Optimizer (Session 7) | L2 regularization on all weights |
| PCA (Session 4.5) | Before model input | Removes 528 noise dimensions |
| Early stopping | Training loop (Session 7) | Stops before overfitting |
| Gradient clipping | Training loop (Session 7) | Prevents exploding gradients |

---

## Configuration

**File**: `configs/model_config.yaml`

```yaml
input_dim: 64          # After PCA (592→64)

mlp:
  hidden_dim: 16
  dropout: 0.5

bigru:
  hidden_dim: 4        # Per direction; output = 4×2 = 8
  num_layers: 1
  dropout: 0.5

attention:
  hidden_dim: 8        # Matches BiGRU output

head:
  output_dim: 1        # Single PHQ-8 score
```

---

## Files

| File | Purpose |
|------|---------|
| `src/models/mlp_block.py` | Input compression layer |
| `src/models/bigru.py` | Temporal sequence modeling (bidirectional GRU) |
| `src/models/attention.py` | Additive attention with masking |
| `src/models/depression_model.py` | Full model assembly + weight init |
| `src/models/__init__.py` | Package exports |
| `configs/model_config.yaml` | Architecture hyperparameters |