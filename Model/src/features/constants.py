"""
Central source of truth for feature dimensions (BP-5).

All feature extractors, fusion, normalizer, and PCA reference these constants
instead of hardcoding dimensions across 5+ files.
"""

EGEMAPS_DIM = 88       # eGeMAPSv02 functionals
MFCC_DIM = 120         # 40 MFCCs × 3 (static + delta + delta-delta)
TEXT_DIM = 384          # SBERT all-MiniLM-L6-v2 embedding
HUBERT_DIM = 768       # HuBERT Base hidden size
FUSED_DIM = EGEMAPS_DIM + MFCC_DIM + TEXT_DIM  # 592
FUSED_V2_DIM = HUBERT_DIM + TEXT_DIM  # 1152
PCA_OUTPUT_DIM = 64    # PCA-reduced model input
