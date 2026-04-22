# Session 7: Training Pipeline

## Purpose

Train the DepressionModel to predict PHQ-8 scores from PCA-reduced features.
Handles the complete training loop with validation, early stopping,
checkpointing, and metric tracking.

---

## Training Flow

```
┌──────────────────────────────────────────────────────────┐
│                    For each epoch:                         │
│                                                           │
│   [Train]  DataLoader → forward → loss → backward → step │
│       │                                                   │
│       ▼                                                   │
│   [Validate]  DataLoader → forward → metrics (CCC, RMSE) │
│       │                                                   │
│       ▼                                                   │
│   [Schedule]  ReduceLROnPlateau(val_CCC)                 │
│       │                                                   │
│       ▼                                                   │
│   [Checkpoint]  Save if val_CCC improved                  │
│       │                                                   │
│       ▼                                                   │
│   [Early Stop?]  patience=5 on val_CCC                   │
│       │                                                   │
│       └── If stop → load best checkpoint → evaluate test  │
└──────────────────────────────────────────────────────────┘
```

---

## Loss Function: Weighted MSE

**File**: `src/training/losses.py`

Standard MSE with upweighting for depressed participants:

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot (\hat{y}_i - y_i)^2$$

Where:

$$w_i = \begin{cases} 2.0 & \text{if } y_i \geq 10 \text{ (depressed)} \\ 1.0 & \text{if } y_i < 10 \text{ (non-depressed)} \end{cases}$$

### Why Weighted?

The DAIC-WOZ dataset is imbalanced:

| Group | Count | Percentage | Weight |
|-------|-------|-----------|--------|
| PHQ < 10 (non-depressed) | ~140 | ~86% | 1.0× |
| PHQ ≥ 10 (depressed) | ~23 | ~14% | 2.0× |

Without weighting, the model could learn to predict ~5 for everyone
(average PHQ-8 score) and achieve low MSE. The 2× weight forces the
model to pay attention to depressed participants.

---

## Metrics

**File**: `src/training/metrics.py`

### Primary: Concordance Correlation Coefficient (CCC)

$$\text{CCC} = \frac{2 \rho \sigma_x \sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}$$

Where:
- $\rho$ — Pearson correlation between predictions and targets
- $\sigma_x, \sigma_y$ — standard deviations
- $\mu_x, \mu_y$ — means

CCC measures **agreement** (not just correlation):
- CCC = 1.0 → perfect agreement
- CCC = 0.0 → no agreement
- CCC < 0 → worse than random

**Why CCC over Pearson $r$?** Pearson only measures linear correlation.
A model predicting $2y + 5$ would have perfect Pearson $r$ but terrible
actual predictions. CCC penalizes both scale and offset differences.

CCC is the **standard metric for AVEC depression challenges** and DAIC-WOZ evaluations.

### Secondary Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| RMSE | $\sqrt{\frac{1}{N}\sum(y - \hat{y})^2}$ | Average prediction error (same units as PHQ-8) |
| MAE | $\frac{1}{N}\sum|y - \hat{y}|$ | Average absolute error (robust to outliers) |

---

## Optimizer & Scheduler

### AdamW Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-3  # L2 regularization
)
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning rate | 1e-3 | Standard starting point for small models |
| Weight decay | 1e-3 | L2 regularization; prevents large weights |
| Betas | (0.9, 0.999) | Default Adam momentum terms |

### ReduceLROnPlateau Scheduler

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',        # Higher CCC is better
    patience=3,        # Wait 3 epochs before reducing
    factor=0.5,        # Halve the learning rate
    min_lr=1e-6        # Don't go below this
)
```

The scheduler monitors **validation CCC**. If CCC doesn't improve for 3 epochs,
the learning rate is halved. This allows fine-grained optimization as the model
converges.

```
Epoch  1-10:  lr = 1e-3  (exploration)
Epoch 11-13:  lr = 1e-3  (CCC plateaus)
Epoch 14:     lr = 5e-4  (scheduler triggers)
Epoch 15-20:  lr = 5e-4  (fine-tuning)
...
```

---

## Early Stopping

**File**: `src/training/early_stopping.py`

Stops training when validation CCC stops improving.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Monitor | Validation CCC | Primary metric |
| Mode | Max (higher is better) | CCC ∈ [-1, 1] |
| Patience | 5 epochs | Enough time to recover from local dips |
| Min delta | 0.0 | Any improvement counts |

```
Epoch 1:  val_CCC = 0.15  → best = 0.15, counter = 0
Epoch 2:  val_CCC = 0.22  → best = 0.22, counter = 0  ← improved
Epoch 3:  val_CCC = 0.20  → counter = 1                ← no improvement
Epoch 4:  val_CCC = 0.19  → counter = 2
Epoch 5:  val_CCC = 0.25  → best = 0.25, counter = 0  ← improved!
Epoch 6:  val_CCC = 0.24  → counter = 1
...
Epoch K:  counter = 5  → STOP. Load best model (epoch 5).
```

When early stopping triggers, it loads the **best model state dict** (saved
internally), not the latest epoch's weights.

---

## Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients in the GRU. Without clipping, a single outlier
batch could produce massive gradients that destabilize all learned weights.

---

## Checkpointing

**File**: `src/training/trainer.py`

### What Gets Saved

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_metric": best_ccc,
    "model_config": model_architecture_config,
    "training_config": training_hyperparameters,
}
```

### Checkpoint Files

| File | When Saved | Purpose |
|------|-----------|---------|
| `checkpoints/best_model.pt` | Val CCC improves | Best model for inference |
| `checkpoints/latest_model.pt` | Every epoch | Resume interrupted training |

### Resume Training

```python
trainer = Trainer(model, config)
trainer.resume_from_checkpoint("checkpoints/latest_model.pt")
trainer.train(train_loader, dev_loader)
```

---

## Reproducibility

All sources of randomness are seeded:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Training History

Per-epoch metrics saved as JSON for later plotting:

```json
{
    "train_loss": [2.5, 2.1, 1.8, ...],
    "val_loss": [2.8, 2.3, 2.0, ...],
    "val_ccc": [0.1, 0.2, 0.25, ...],
    "val_rmse": [7.0, 6.5, 6.2, ...],
    "val_mae": [5.5, 5.0, 4.8, ...],
    "learning_rates": [1e-3, 1e-3, 5e-4, ...]
}
```

---

## Configuration

**File**: `configs/training_config.yaml`

```yaml
optimizer:
  type: "adamw"
  lr: 0.001
  weight_decay: 0.001

scheduler:
  type: "reduce_on_plateau"
  mode: "max"
  patience: 3
  factor: 0.5
  min_lr: 0.000001

loss:
  type: "weighted_mse"
  threshold: 10
  high_weight: 2.0

early_stopping:
  patience: 5
  mode: "max"
  min_delta: 0.0

training:
  max_epochs: 100
  gradient_clip_norm: 1.0
  seed: 42

checkpoint:
  save_dir: "checkpoints"
  save_best: true
  save_latest: true
```

---

## Running Training

```bash
# Full pipeline: extract → train
python scripts/extract_features.py --config configs/feature_config.yaml
python scripts/train.py --config configs/training_config.yaml

# Output:
#   checkpoints/best_model.pt           ← use this for inference
#   checkpoints/latest_model.pt         ← resume training
#   checkpoints/scalers/feature_scalers.pkl
#   checkpoints/scalers/pca_reducer.pkl
#   logs/training.log                   ← full training log
```

---

## Files

| File | Purpose |
|------|---------|
| `src/training/trainer.py` | Main training loop + checkpoint management |
| `src/training/losses.py` | WeightedMSE loss function |
| `src/training/metrics.py` | CCC, RMSE, MAE computation |
| `src/training/early_stopping.py` | Patience-based early stopping |
| `src/training/__init__.py` | Package exports |
| `scripts/train.py` | CLI entry point for training |
| `configs/training_config.yaml` | All training hyperparameters |