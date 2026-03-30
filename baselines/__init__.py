"""
Baseline models for OOS detection.

Simple models without specialized OOS support:
- TfidfLogreg: TF-IDF + Logistic Regression (OOS as extra class)
- EmbeddingThreshold: Cosine similarity threshold
- ZeroShotLLM: LLM-based zero-shot classification
"""

from .tfidf_logreg import TfidfLogreg
from .embedding_threshold import EmbeddingThreshold

__all__ = [
    "TfidfLogreg",
    "EmbeddingThreshold",
]
