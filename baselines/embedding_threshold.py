"""
Embedding-based OOS detection with cosine similarity threshold.

Simple approach: compute cosine similarity to training examples,
if max similarity is below threshold — classify as OOS.
"""

from __future__ import annotations
import numpy as np


class EmbeddingThreshold:
    """
    OOS detection via cosine similarity threshold.

    For each test sample:
    1. Compute embedding
    2. Find max cosine similarity to any training sample
    3. If max_sim < threshold -> OOS

    Attributes:
        model_name: sentence-transformers model name
        threshold: similarity threshold for OOS detection
        train_embeddings: cached training embeddings
        train_labels: training labels
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float | None = None,
        threshold_percentile: float = 95,
    ):
        """
        Initialize embedding threshold model.

        Args:
            model_name: sentence-transformers model to use
            threshold: fixed similarity threshold (if None, use percentile)
            threshold_percentile: percentile of train similarities for threshold
        """
        self.model_name = model_name
        self.threshold = threshold
        self.threshold_percentile = threshold_percentile
        self.encoder = None
        self.train_embeddings = None
        self.train_labels = None

    def fit(self, texts: list[str], labels: list[int]) -> "EmbeddingThreshold":
        """
        Fit the model (compute and cache training embeddings).

        Args:
            texts: list of text samples (in-scope only for fitting)
            labels: list of integer labels

        Returns:
            self
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Predict class labels (including OOS = -1).

        Args:
            texts: list of text samples

        Returns:
            predicted labels (-1 for OOS)
        """
        # TODO: реализовать
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Get OOS probability (1 - max_similarity).

        Args:
            texts: list of text samples

        Returns:
            OOS probability for each sample
        """
        # TODO: реализовать
        raise NotImplementedError

    def get_oos_scores(self, texts: list[str]) -> np.ndarray:
        """
        Get OOS scores (1 - max_similarity).

        Higher score = more likely OOS.

        Args:
            texts: list of text samples

        Returns:
            OOS score for each sample
        """
        # TODO: реализовать
        raise NotImplementedError

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for texts."""
        # TODO: реализовать
        raise NotImplementedError

    def _compute_similarities(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Compute max similarity to training set for each query."""
        # TODO: реализовать
        raise NotImplementedError
