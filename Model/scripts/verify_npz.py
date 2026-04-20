#!/usr/bin/env python3
"""Verify extracted NPZ files have correct shapes and dtypes."""

import sys
from pathlib import Path
import numpy as np

NPZ_DIR = Path("data/features")
REQUIRED_KEYS = {"egemaps", "mfcc", "text_embeddings"}
EXPECTED_DIMS = {"egemaps": 88, "mfcc": 120, "text_embeddings": 384}


def main():
    npz_files = sorted(NPZ_DIR.glob("*_training.npz"))
    if not npz_files:
        print(f"ERROR: No NPZ files found in {NPZ_DIR}")
        sys.exit(1)

    print(f"Found {len(npz_files)} NPZ files in {NPZ_DIR}\n")

    errors = []
    for f in npz_files:
        data = np.load(f)
        keys = set(data.keys())

        missing = REQUIRED_KEYS - keys
        if missing:
            errors.append(f"{f.name}: missing keys {missing}")
            continue

        n_chunks = None
        for key in REQUIRED_KEYS:
            arr = data[key]
            if arr.dtype != np.float32:
                errors.append(f"{f.name}: {key} dtype={arr.dtype} (expected float32)")
            if arr.ndim != 2:
                errors.append(f"{f.name}: {key} ndim={arr.ndim} (expected 2)")
                continue
            if arr.shape[1] != EXPECTED_DIMS[key]:
                errors.append(f"{f.name}: {key} dim={arr.shape[1]} (expected {EXPECTED_DIMS[key]})")
            if n_chunks is None:
                n_chunks = arr.shape[0]
            elif arr.shape[0] != n_chunks:
                errors.append(f"{f.name}: {key} has {arr.shape[0]} chunks (expected {n_chunks})")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        # Print summary
        sample = np.load(npz_files[0])
        print("All files OK!")
        print(f"  Files: {len(npz_files)}")
        print(f"  Keys: {sorted(REQUIRED_KEYS)}")
        print(f"  Sample shapes: {', '.join(f'{k}={sample[k].shape}' for k in sorted(REQUIRED_KEYS))}")
        print(f"  Feature total: {sum(EXPECTED_DIMS.values())} dims")


if __name__ == "__main__":
    main()
