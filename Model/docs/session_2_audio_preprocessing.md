# Session 2: Audio Preprocessing Pipeline

## Purpose

Convert raw audio files into clean, uniform chunks ready for feature extraction.
This is the first stage of the inference pipeline — garbage in, garbage out.

---

## Pipeline Flow

```
Raw Audio (.wav/.mp3/.flac)
    │
    ▼
[1. Load & Resample]  → mono, 16kHz, float32
    │
    ▼
[2. Voice Activity Detection]  → remove silence/noise
    │
    ▼
[3. Chunking]  → fixed 5-second segments with overlap
    │
    ▼
Output: List[AudioChunk]  → (C, 80000) numpy arrays
```

---

## Components

### 1. Audio Loading & Resampling

**File**: `src/preprocessing/audio_preprocessor.py`

| Parameter | Value | Reason |
|-----------|-------|--------|
| Target sample rate | 16,000 Hz | Standard for speech processing; matches OpenSMILE/Whisper |
| Channels | Mono | Depression cues are in voice, not spatial audio |
| Format | float32 [-1, 1] | Normalized for consistent feature extraction |

```
Input:  any_audio.wav (44.1kHz, stereo, 16-bit)
Output: np.ndarray (N_samples,) at 16kHz mono float32
```

### 2. Voice Activity Detection (VAD)

**File**: `src/preprocessing/vad.py`

Energy-based VAD that removes silence and background noise.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Frame length | 25 ms (400 samples) | Standard speech frame |
| Hop length | 10 ms (160 samples) | Standard speech hop |
| Energy threshold | Adaptive (mean - 1 std) | Robust to recording conditions |
| Min speech duration | 0.3 seconds | Avoid spurious detections |

```
Input:  audio with silence gaps (e.g., interviewer pauses)
Output: concatenated speech-only segments
```

**Fallback**: If VAD removes ALL audio (extremely quiet recording),
returns the original audio unchanged. This prevents empty chunks downstream.

### 3. Chunking

**File**: `src/preprocessing/chunker.py`

Splits continuous speech into fixed-length overlapping windows.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Chunk duration | 5.0 seconds | Captures prosodic patterns; enough for MFCC statistics |
| Overlap | 25% (1.25 seconds) | Prevents information loss at boundaries |
| Min chunk duration | 1.0 second | Reject fragments too short for features |
| Samples per chunk | 80,000 (5s × 16kHz) | Fixed size for batch processing |

```
Input:  speech audio (variable length)
Output: List[AudioChunk] where each chunk is (80000,) numpy array
        + chunk_start_times for temporal alignment
```

**Padding**: Final chunk is zero-padded to 80,000 samples if shorter than 5 seconds.

---

## AudioChunk Dataclass

```python
@dataclass
class AudioChunk:
    audio: np.ndarray      # (80000,) float32 at 16kHz
    start_time: float      # seconds from original audio start
    end_time: float        # seconds from original audio start
    chunk_index: int       # 0-based index
    sample_rate: int       # always 16000
```

---

## Full Preprocessing Entry Point

```python
preprocessor = AudioPreprocessor(config)

# Training: already have CSVs, skip preprocessing
# Inference: process raw audio
result = preprocessor.preprocess_for_inference("patient_audio.wav")

result["chunks"]             # np.ndarray (C, 80000)
result["chunk_start_times"]  # List[float], length C
result["sample_rate"]        # 16000
```

---

## Configuration

**File**: `configs/audio_config.yaml`

```yaml
sample_rate: 16000
chunk_duration: 5.0
chunk_overlap: 0.25
vad:
  frame_length_ms: 25
  hop_length_ms: 10
  min_speech_duration: 0.3
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/preprocessing/audio_preprocessor.py` | ~200 | Main entry point: load → VAD → chunk |
| `src/preprocessing/vad.py` | ~100 | Energy-based voice activity detection |
| `src/preprocessing/chunker.py` | ~120 | Fixed-length overlapping window chunking |
| `src/preprocessing/__init__.py` | ~10 | Package exports |
| `configs/audio_config.yaml` | ~15 | All preprocessing parameters |

---

## Why These Choices?

| Choice | Alternatives Considered | Why We Chose This |
|--------|------------------------|-------------------|
| 16kHz | 8kHz, 22.05kHz, 44.1kHz | Industry standard for speech; Whisper/OpenSMILE expect it |
| 5s chunks | 3s, 10s, variable | 5s captures prosodic contours; short enough for many chunks per participant |
| 25% overlap | 0%, 50% | Balance between coverage and computational cost |
| Energy VAD | webrtcvad, Silero VAD | Zero dependencies; sufficient for clinical recordings (quiet rooms) |
| Zero-pad short chunks | Discard | Preserves terminal speech segments which may contain depression cues |