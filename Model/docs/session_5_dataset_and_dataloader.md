# Session 5: Dataset & DataLoader

## Purpose

Convert per-participant PCA-reduced features into batched, padded tensors
with attention masks — the format our model expects.

---

## The Problem

Each participant has a different number of audio chunks (time-steps):

| Participant | Chunks | Feature Shape |
|-------------|--------|---------------|
| P300 | 45 | (45, 64) |
| P301 | 120 | (120, 64) |
| P302 | 78 | (78, 64) |

PyTorch requires tensors of the same shape within a batch.
We need padding + masking.

---

## Data Flow

```
Per-participant features (T_i, 64)
    │
    ▼
DepressionDataset  →  (features, label, participant_id)
    │
    ▼
collate_fn  →  pad to max length in batch + create masks
    │
    ▼
DataLoader  →  (B, T_max, 64) + mask(B, T_max) + lengths(B,) + labels(B,)
```

---

## DepressionDataset

**File**: `src/dataset/depression_dataset.py`

A PyTorch `Dataset` that returns one participant's data per index.

```python
# Training: load from pre-extracted features
dataset = DepressionDataset.from_split_csv(
    feature_dict={"300": (45, 64), "301": (120, 64), ...},
    labels_dict={"300": 12.0, "301": 3.0, ...}
)

# Inference: load from features directly
dataset = DepressionDataset.for_inference(features)  # (C, 64)
```

### What `__getitem__` Returns

```python
{
    "features": torch.Tensor,      # (T_i, 64) — variable length
    "label": torch.Tensor,         # scalar PHQ-8 score
    "participant_id": str           # e.g., "300"
}
```

---

## Collation (Padding + Masking)

**File**: `src/dataset/collate.py`

When the DataLoader combines multiple samples into a batch, the `collate_fn`
handles variable-length sequences:

### Step 1: Sort by Length (Descending)

```
Before sorting:  [T=45, T=120, T=78]
After sorting:   [T=120, T=78, T=45]
```

Sorting enables `pack_padded_sequence` in the BiGRU (with `enforce_sorted=True`),
which skips computation on padded positions.

### Step 2: Pad to Max Length

```
T_max = 120 (longest in batch)

P301: [■■■■■■■■■■■■■■■■■■■■] (120)  → no padding
P302: [■■■■■■■■■■■░░░░░░░░░] (78)   → 42 zeros appended
P300: [■■■■■■■■░░░░░░░░░░░░] (45)   → 75 zeros appended
```

### Step 3: Create Attention Mask

```
mask[0] = [True  True  True  ... True  True  True ]  (all 120 valid)
mask[1] = [True  True  True  ... True  False False]  (78 valid, 42 padded)
mask[2] = [True  True  True  ... False False False]  (45 valid, 75 padded)
```

The mask tells the attention layer: **only attend to `True` positions**.
Padded positions get `-inf` attention scores → 0 after softmax.

### Collate Output

```python
{
    "features": torch.Tensor,  # (B, T_max, 64) — zero-padded
    "labels": torch.Tensor,    # (B,) — PHQ-8 scores
    "mask": torch.Tensor,      # (B, T_max) — True=valid, False=pad
    "lengths": torch.Tensor    # (B,) — original lengths [120, 78, 45]
}
```

---

## SequenceBuilder (DataLoader Factory)

**File**: `src/dataset/sequence_builder.py`

Builds DataLoaders with appropriate settings for each context:

| Context | Batch Size | Shuffle | Drop Last | Workers |
|---------|-----------|---------|-----------|---------|
| Training | 8 | ✅ Yes | ✅ Yes | 0 |
| Evaluation (dev/test) | 8 | ❌ No | ❌ No | 0 |
| Inference | 1 | ❌ No | ❌ No | 0 |

```python
builder = SequenceBuilder(batch_size=8)

train_loader = builder.build_train_loader(train_dataset)
dev_loader = builder.build_eval_loader(dev_dataset)
inference_loader = builder.build_inference_loader(inference_dataset)
```

**Why `drop_last=True` for training?** The last batch may have fewer samples
(e.g., 3 instead of 8), which can cause unstable gradients with BatchNorm
or affect loss weighting. Dropping it ensures consistent batch sizes.

**Why `num_workers=0`?** On small datasets, the overhead of multiprocess
data loading exceeds the benefit. Single-process is faster.

---

## Configuration

**File**: `configs/dataset_config.yaml`

```yaml
train:
  batch_size: 8
  shuffle: true
  drop_last: true
eval:
  batch_size: 8
  shuffle: false
  drop_last: false
num_workers: 0
```

---

## Files

| File | Purpose |
|------|---------|
| `src/dataset/depression_dataset.py` | PyTorch Dataset for train/eval/inference |
| `src/dataset/collate.py` | Padding, masking, length-sorting collation |
| `src/dataset/sequence_builder.py` | DataLoader factory for all contexts |
| `src/dataset/__init__.py` | Package exports |
| `configs/dataset_config.yaml` | Batch size, shuffle, worker settings |