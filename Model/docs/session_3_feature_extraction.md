# Session 3: Multi-Modal Feature Extraction

## Purpose

Extract three complementary feature modalities from speech that capture
different aspects of depression:

| Modality | Extractor | Dimensions | What It Captures |
|----------|-----------|------------|-----------------|
| **eGeMAPS** | OpenSMILE | 88 | Prosody, voice quality, formants, jitter, shimmer |
| **MFCC** | librosa | 120 | Spectral envelope, articulation patterns |
| **Text** | Whisper + SBERT | 384 | Linguistic content, sentiment, cognitive patterns |

**Total: 592 features per time-step**

---

## Pipeline Flow

```
Audio Chunk (80000 samples, 16kHz)
    │
    ├──→ [OpenSMILE eGeMAPS]  → (88,)   acoustic/prosodic
    │
    ├──→ [librosa MFCC+Δ+ΔΔ] → (120,)  spectral
    │
    └──→ [Whisper → SBERT]    → (384,)  linguistic
    
Per participant: C chunks → (C, 88), (C, 120), (C, 384)
```

---

## Modality 1: eGeMAPS (88 features)

**File**: `src/features/egemaps_extractor.py`

The extended Geneva Minimalistic Acoustic Parameter Set — a standardized
feature set designed for affective computing research.

### Feature Groups

| Group | Count | Examples | Depression Relevance |
|-------|-------|---------|---------------------|
| Frequency (F0) | ~10 | Pitch mean, std, range | Monotone speech → depression |
| Energy/Amplitude | ~8 | Loudness, shimmer | Reduced energy → depression |
| Spectral | ~20 | Spectral slope, formant frequencies | Voice quality changes |
| Temporal | ~6 | Speech rate, pause duration | Psychomotor retardation |
| Voice quality | ~15 | Jitter, HNR, CPP | Breathiness, hoarseness |
| Formants (F1-F3) | ~18 | Bandwidth, frequency | Articulatory precision |
| MFCC (built-in) | ~11 | Cepstral coefficients | Redundant with MFCC modality (complementary stats) |

### Usage

```python
# Training: from pre-extracted CSV
extractor = EgemapsExtractor(config)
features = extractor.extract(csv_path="path/to/egemaps.csv")  # (N, 88)

# Inference: from audio chunks
features = extractor.extract_from_audio(chunks, sample_rate=16000)  # (C, 88)
```

---

## Modality 2: MFCC + Deltas (120 features)

**File**: `src/features/mfcc_extractor.py`

Mel-Frequency Cepstral Coefficients capture the spectral envelope of speech,
which reflects articulatory and phonetic characteristics.

### Feature Breakdown

| Feature | Count | Formula | What It Captures |
|---------|-------|---------|-----------------|
| MFCC (1-40) | 40 | DCT of log-mel spectrogram | Spectral envelope shape |
| Δ-MFCC | 40 | First derivative of MFCC | Rate of spectral change |
| ΔΔ-MFCC | 40 | Second derivative of MFCC | Acceleration of change |

**Per chunk**: Compute frame-level MFCCs → aggregate with **mean + std** → 40×3 = 120 features

| Statistic | Count | Purpose |
|-----------|-------|---------|
| Mean of 40 MFCCs | 40 | Average spectral shape |
| Mean of 40 Δ-MFCCs | 40 | Average rate of change |
| Mean of 40 ΔΔ-MFCCs | 40 | Average acceleration |

### Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_mfcc | 40 | Captures fine spectral detail |
| n_fft | 512 (32ms at 16kHz) | Standard speech analysis window |
| hop_length | 160 (10ms) | Standard speech hop |
| n_mels | 64 | Sufficient for speech band (0-8kHz) |
| fmin | 50 Hz | Below F0 range |
| fmax | 8000 Hz | Nyquist of 16kHz input |

### Usage

```python
# Training: from pre-extracted CSV
features = extractor.extract(csv_path="path/to/mfcc.csv")  # (N, 120)

# Inference: from audio chunks
features = extractor.extract_from_audio(chunks, sample_rate=16000)  # (C, 120)
```

---

## Modality 3: Text Embeddings (384 features)

**File**: `src/features/text_extractor.py`

Speech-to-text transcription followed by semantic embedding.
Captures what the person says, not just how they say it.

### Two-Stage Pipeline

```
Audio Chunk → [Whisper Base] → transcript text → [SBERT] → (384,) embedding
```

| Stage | Model | Output | Purpose |
|-------|-------|--------|---------|
| ASR | `openai/whisper-base` | Text string | Transcribe speech to text |
| Embedding | `all-MiniLM-L6-v2` | (384,) vector | Semantic representation |

### Why These Models?

| Model | Size | Alternative | Why This One |
|-------|------|-------------|-------------|
| Whisper Base | 74M params | Whisper Large (1.5B) | Balance of accuracy vs. speed; sufficient for English clinical interviews |
| SBERT MiniLM | 22M params | SBERT Large (330M) | 384-dim output is compact; excellent sentence-level semantics |

### Depression-Relevant Linguistic Patterns

| Pattern | Example | PHQ Correlation |
|---------|---------|----------------|
| Negative affect words | "hopeless", "worthless" | Strong positive |
| First-person pronouns | "I feel", "I can't" | Moderate positive |
| Absolute language | "always", "never", "nothing" | Moderate positive |
| Reduced vocabulary | Short, simple responses | Moderate positive |
| Past tense focus | "I used to..." | Weak positive |

SBERT embeddings implicitly capture these patterns in the 384-dim space.

### Usage

```python
# Training: from pre-extracted transcript CSV or text
features = extractor.extract(csv_path="path/to/text.csv")  # (N, 384)

# Inference: from audio (runs Whisper + SBERT)
features = extractor.extract_from_audio(
    audio_chunks=chunks, sample_rate=16000
)  # (C, 384)
```

---

## Feature Storage

**File**: `src/features/feature_store.py`

Saves/loads extracted features as `.npz` files per participant.

```
data/processed/features/
├── 300/
│   └── features.npz    # keys: "egemaps", "mfcc", "text_embeddings"
├── 301/
│   └── features.npz
└── ...
```

```python
store = FeatureStore(base_dir="data/processed/features")

# Save
store.save("300", {
    "egemaps": np.array(...),       # (T, 88)
    "mfcc": np.array(...),          # (T, 120)
    "text_embeddings": np.array(...)  # (T, 384)
})

# Load
features = store.load("300")
features["egemaps"].shape  # (T, 88)
```

---

## Configuration

**File**: `configs/feature_config.yaml`

```yaml
egemaps:
  feature_set: "eGeMAPSv02"
  feature_level: "functionals"
  output_dim: 88

mfcc:
  n_mfcc: 40
  n_fft: 512
  hop_length: 160
  n_mels: 64
  include_deltas: true
  output_dim: 120

text:
  whisper_model: "openai/whisper-base"
  sbert_model: "sentence-transformers/all-MiniLM-L6-v2"
  output_dim: 384
```

---

## Files

| File | Purpose |
|------|---------|
| `src/features/egemaps_extractor.py` | OpenSMILE eGeMAPS extraction |
| `src/features/mfcc_extractor.py` | librosa MFCC + Δ + ΔΔ extraction |
| `src/features/text_extractor.py` | Whisper ASR + SBERT embedding |
| `src/features/feature_store.py` | Save/load .npz per participant |
| `src/features/__init__.py` | Package exports |
| `configs/feature_config.yaml` | All feature parameters |