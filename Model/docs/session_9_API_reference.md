# Session 9: REST API (FastAPI)

## Purpose

Expose the inference pipeline as HTTP endpoints for web/mobile integration.
Accepts audio file uploads and returns PHQ-8 predictions as JSON.

---

## Endpoints

### `POST /predict` — Single Prediction

Upload one audio file, get one prediction.

**Request:**
```
POST /predict
Content-Type: multipart/form-data

file: <audio_file.wav>
```

**Response (200):**
```json
{
    "participant_id": "uploaded_audio",
    "phq8_score": 12.3,
    "severity": "moderate",
    "num_chunks": 15,
    "processing_time_seconds": 2.69,
    "timestamp": "2025-01-15T14:30:00.000Z"
}
```

**Error Responses:**

| Code | Reason | Body |
|------|--------|------|
| 422 | Invalid file extension | `{"detail": "Invalid file type. Allowed: .wav, .mp3, ..."}` |
| 413 | File too large (>100MB) | `{"detail": "File too large. Max: 100 MB"}` |
| 503 | Model not loaded | `{"detail": "Model not loaded. Check server logs."}` |
| 500 | Processing error | `{"detail": "Prediction failed: <error>"}` |

---

### `POST /predict/batch` — Batch Prediction

Upload multiple audio files, get predictions for each.

**Request:**
```
POST /predict/batch
Content-Type: multipart/form-data

files: <audio1.wav>
files: <audio2.wav>
files: <audio3.mp3>
```

**Response (200):**
```json
{
    "predictions": [
        {
            "participant_id": "audio1.wav",
            "phq8_score": 12.3,
            "severity": "moderate",
            "num_chunks": 15,
            "processing_time_seconds": 2.69,
            "timestamp": "..."
        },
        {
            "participant_id": "audio2.wav",
            "phq8_score": 3.1,
            "severity": "minimal",
            "num_chunks": 8,
            "processing_time_seconds": 1.5,
            "timestamp": "..."
        }
    ],
    "failed": [
        {
            "filename": "audio3.mp3",
            "error": "Preprocessing failed: corrupt file"
        }
    ],
    "total_files": 3,
    "successful": 2,
    "processing_time_seconds": 4.19
}
```

**Limits:**
- Max 20 files per batch (configurable in `inference_config.yaml`)
- Each file subject to same validation as single prediction

---

### `GET /health` — Health Check

**Response (200):**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda",
    "timestamp": "2025-01-15T14:30:00.000Z"
}
```

If the model failed to load during startup:
```json
{
    "status": "degraded",
    "model_loaded": false,
    "device": "cpu",
    "timestamp": "..."
}
```

---

## Architecture

```
Client (browser/app)
    │
    ▼ HTTP
┌─────────────────────┐
│   FastAPI (uvicorn)  │
│                      │
│  ┌─────────┐         │
│  │ routes  │         │  ← validates files, handles errors
│  └────┬────┘         │
│       │              │
│  ┌────▼──────────┐   │
│  │ InferencePipeline │  ← loaded on startup (lifespan)
│  └────┬──────────┘   │
│       │              │
│  ┌────▼──────────┐   │
│  │ ExperimentTracker │  ← logs predictions to SQLite (optional)
│  └───────────────┘   │
└─────────────────────┘
```

---

## File Upload Security

| Check | Implementation | Reason |
|-------|---------------|--------|
| Extension whitelist | `.wav, .mp3, .flac, .ogg, .m4a, .webm` | Reject non-audio files |
| Content-type check | MIME type validation | Defense against renamed files |
| File size limit | 100 MB (streaming check) | Prevent memory exhaustion |
| Temp file cleanup | `finally` block with `os.path.exists()` | No disk leaks |
| Safe temp paths | `tempfile.mkstemp()` | No race conditions |

---

## Startup (Lifespan)

When the FastAPI app starts, it loads the full inference pipeline:

```python
@asynccontextmanager
async def lifespan(app):
    # Load pipeline (model + extractors + scalers + PCA)
    pipeline = InferencePipeline(config_path)
    app.state.pipeline = pipeline

    # Load database tracker (optional, non-blocking)
    tracker = ExperimentTracker(db_url)
    app.state.tracker = tracker

    yield  # App is running

    # Cleanup on shutdown
```

If pipeline loading fails, the app still starts but `/predict` returns 503.

---

## Database Integration

Every prediction is logged to SQLite (if available):

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    participant_id TEXT,
    phq8_score REAL,
    severity TEXT,
    num_chunks INTEGER,
    inference_time_seconds REAL,
    device TEXT,
    created_at TIMESTAMP
);
```

Database errors are **non-blocking** — if the DB is down, predictions still work.

---

## Running the API

**File**: `scripts/serve.py`

```bash
# Default: localhost:8000
python scripts/serve.py

# Custom host/port
python scripts/serve.py --host 0.0.0.0 --port 8080

# Custom config
python scripts/serve.py --config custom_inference.yaml

# OpenAPI docs: http://localhost:8000/docs
```

---

## Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@patient_audio.wav"

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav"
```

---

## Configuration

**File**: `configs/inference_config.yaml`

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  max_file_size_mb: 100
  max_batch_size: 20
  allowed_extensions: [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"]
  cors_origins: ["*"]       # Restrict in production!
```

---

## Files

| File | Purpose |
|------|---------|
| `src/api/app.py` | FastAPI app factory with lifespan loading |
| `src/api/routes.py` | Endpoint definitions + file validation |
| `src/api/schemas.py` | Pydantic request/response models |
| `src/api/__init__.py` | Package exports |
| `scripts/serve.py` | CLI entry point (uvicorn launcher) |
| `configs/inference_config.yaml` | API + model + artifact configuration |