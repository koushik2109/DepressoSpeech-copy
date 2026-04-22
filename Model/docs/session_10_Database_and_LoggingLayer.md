# Session 10: Database & Logging Layer

## Purpose

Track all experiments, predictions, and system events with structured logging
to files and a SQLite database. Essential for reproducibility, debugging,
and clinical audit trails.

---

## Part 1: Centralized Logging

### The Problem

Without centralized logging, each module prints to console with inconsistent
formats. Logs are lost when the terminal closes. No way to debug production
issues after the fact.

### Solution: File + Console Handlers

**File**: `src/utils/logging_config.py`

```python
from src.utils import setup_logging

# Called once at script entry point
setup_logging(log_dir="logs", task_name="training")
# Creates: logs/training.log
```

All modules use their own logger: `logger = logging.getLogger(__name__)`.
The centralized setup captures all of them.

### Log Format

**Console** (INFO and above):
```
2025-01-15 14:30:00 | INFO | Training epoch 1/100
2025-01-15 14:30:05 | INFO | [CHECKPOINT] Saved best model: CCC=0.35
```

**File** (DEBUG and above):
```
2025-01-15 14:30:00 | DEBUG | src.training.trainer | Batch 1/20: loss=2.534
2025-01-15 14:30:00 | INFO  | src.training.trainer | Training epoch 1/100
2025-01-15 14:30:05 | INFO  | src.training.trainer | [CHECKPOINT] Saved best model
```

### Log Files by Task

| Script | Log File | Contents |
|--------|----------|---------|
| `scripts/extract_features.py` | `logs/feature_extraction.log` | Per-participant extraction progress |
| `scripts/train.py` | `logs/training.log` | Epoch losses, metrics, checkpoints |
| `scripts/predict.py` | `logs/inference.log` | Per-file prediction details |
| `scripts/serve.py` | `logs/api_server.log` | Request handling, errors |

### Log Prefixes

Standardized prefixes for easy grep:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `[CHECKPOINT]` | Model/scaler save/load | `[CHECKPOINT] Saved best_model.pt` |
| `[TRAINING_PATH]` | Training-specific operation | `[TRAINING_PATH] PCA fitted: 592→64` |
| `[INFERENCE_PATH]` | Inference-specific operation | `[INFERENCE_PATH] PCA loaded` |
| `[DATA_FLOW]` | Dimension/shape information | `[DATA_FLOW] PCA transform: (100,592)→(100,64)` |
| `[VALIDATION_CHECK]` | Input validation | `[VALIDATION_CHECK] Expected dim 592, got 384` |
| `[LAYER_START]` | Module initialization | `[LAYER_START] BiGRUEncoder: 16→8` |

```bash
# Find all checkpoint events
grep "\[CHECKPOINT\]" logs/training.log

# Find all dimension changes
grep "\[DATA_FLOW\]" logs/feature_extraction.log
```

---

## Part 2: Run Manager

**File**: `src/utils/run_manager.py`

Manages per-run artifacts and creates summary files.

### Config Snapshot

Before training starts, ALL configuration is saved:

```python
run_manager = RunManager(log_dir="logs")
run_manager.save_config_snapshot({
    "training": training_config,
    "model": model_config,
    "features": feature_config
})
# Creates: logs/config_snapshot.json
```

This ensures you can always reproduce a training run — even if you've
since changed the config files.

### Training Summary

After training completes:

```python
run_manager.save_training_summary(
    best_metrics={"ccc": 0.45, "rmse": 6.2},
    total_epochs=42,
    stop_reason="early_stopping",
    training_time=1800  # seconds
)
# Creates: logs/training_summary.json
```

```json
{
    "best_metrics": {"ccc": 0.45, "rmse": 6.2, "mae": 4.8},
    "total_epochs": 42,
    "stop_reason": "early_stopping",
    "training_time_seconds": 1800,
    "training_time_human": "30.0m",
    "timestamp": "2025-01-15T15:00:00"
}
```

---

## Part 3: SQLite Database

### Schema

**File**: `src/db/models.py`

4 tables tracking experiments and predictions:

```
┌──────────────┐     ┌──────────────────┐
│  Experiment   │     │  TrainingCurve    │
├──────────────┤     ├──────────────────┤
│ id (PK)      │◄───┤│ experiment_id(FK) │
│ name         │     │ epoch             │
│ config (JSON)│     │ train_loss        │
│ status       │     │ val_loss          │
│ best_ccc     │     │ val_ccc           │
│ best_rmse    │     │ learning_rate     │
│ total_epochs │     │ timestamp         │
│ created_at   │     └──────────────────┘
└──────────────┘

┌──────────────┐     ┌──────────────────┐
│ ModelVersion  │     │   Prediction      │
├──────────────┤     ├──────────────────┤
│ id (PK)      │     │ id (PK)          │
│ version      │     │ participant_id    │
│ checkpoint   │     │ phq8_score        │
│ val_ccc      │     │ severity          │
│ val_rmse     │     │ num_chunks        │
│ is_production│     │ inference_time    │
│ created_at   │     │ device            │
└──────────────┘     │ created_at        │
                      └──────────────────┘
```

### Database Location

Default: `data/depresso.db` (SQLite, zero configuration)

For production, swap to PostgreSQL by changing one line in `configs/db_config.yaml`:

```yaml
# Development (SQLite)
database_url: "sqlite:///data/depresso.db"

# Production (PostgreSQL)
database_url: "postgresql://user:pass@host:5432/depresso"
```

### ExperimentTracker

**File**: `src/utils/experiment_tracker.py`

High-level API wrapping all database operations:

```python
tracker = ExperimentTracker(db_url="sqlite:///data/depresso.db")

# Training: track experiment
exp_id = tracker.create_experiment("run_001", config_dict)
tracker.log_epoch(exp_id, epoch=1, train_loss=2.5, val_ccc=0.15)
tracker.finish_experiment(exp_id, best_ccc=0.45, status="completed")

# Inference: track predictions
tracker.log_prediction(
    participant_id="patient_001",
    phq8_score=12.3,
    severity="moderate",
    num_chunks=15,
    inference_time=2.69,
    device="cuda"
)
```

### Non-Blocking Design

All database operations are wrapped in try/except:

```python
try:
    tracker.log_prediction(...)
except Exception as e:
    logger.warning(f"DB logging failed: {e}")
    # Prediction still returned to user
```

**The API and training pipeline NEVER fail due to database errors.**

---

## Configuration

**File**: `configs/db_config.yaml`

```yaml
database_url: "sqlite:///data/depresso.db"
pool_size: 5
max_overflow: 10
echo: false            # Set true to log SQL queries
```

**File**: `configs/logging_config.yaml`

```yaml
log_dir: "logs"
console_level: "INFO"
file_level: "DEBUG"
format: "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
date_format: "%Y-%m-%d %H:%M:%S"

prefixes:
  checkpoint: "[CHECKPOINT]"
  training: "[TRAINING_PATH]"
  inference: "[INFERENCE_PATH]"
  data_flow: "[DATA_FLOW]"
  validation: "[VALIDATION_CHECK]"
  layer: "[LAYER_START]"
```

---

## Files

| File | Purpose |
|------|---------|
| `src/utils/logging_config.py` | Centralized logging setup |
| `src/utils/run_manager.py` | Config snapshots + training summaries |
| `src/utils/experiment_tracker.py` | Database operations wrapper |
| `src/utils/__init__.py` | Package exports |
| `src/db/database.py` | SQLAlchemy engine + session factory |
| `src/db/models.py` | ORM table definitions |
| `src/db/__init__.py` | Package exports |
| `configs/db_config.yaml` | Database connection settings |
| `configs/logging_config.yaml` | Logging format + levels |

---

## Folder Structure After Session 10

```
logs/
├── training.log              ← created by scripts/train.py
├── feature_extraction.log    ← created by scripts/extract_features.py
├── inference.log             ← created by scripts/predict.py
├── api_server.log            ← created by scripts/serve.py
├── config_snapshot.json      ← saved before training
└── training_summary.json     ← saved after training

checkpoints/
├── best_model.pt             ← best validation CCC
├── latest_model.pt           ← resume training
└── scalers/
    ├── feature_scalers.pkl   ← normalizer state
    └── pca_reducer.pkl       ← PCA state

data/
└── depresso.db               ← SQLite database
```