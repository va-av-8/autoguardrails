"""
Base class for binary classification AutoML wrappers (Jailbreak Detection).

Unlike OOS detection (multiclass + threshold on 1-max(P)), binary classification
trains on both classes and uses P(jailbreak) directly as the score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..metrics import f1_oos

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class BinaryFrameworkWrapper(ABC):
    """
    Common interface for binary classification framework wrappers.

    Key differences from OOS BaseFrameworkWrapper:
    - No train filtering: fit() uses all samples (both safe=0 and jailbreak=1)
    - predict_proba() returns P(jailbreak) directly (not 1 - max(P))
    - No _predict_in_domain() needed — direct binary classification
    - Threshold calibration is optional, default_threshold=0.5 used by default
    """

    positive_label: int = 1  # jailbreak

    def __init__(
        self,
        model_name: str,
        default_threshold: float = 0.5,
        embedder: "SentenceTransformer | None" = None,
    ):
        self.model_name = model_name
        self.default_threshold = default_threshold
        self.threshold_: float | None = None
        self.embedder = embedder

    @abstractmethod
    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        precomputed_embeddings: np.ndarray | None = None,
    ) -> "BinaryFrameworkWrapper":
        """
        Fit wrapper on training data.

        Unlike OOS wrappers, does NOT filter out any class.
        Trains binary classifier on both safe (0) and jailbreak (1) samples.

        Args:
            train_texts: List of text samples
            train_labels: List of labels (0=safe, 1=jailbreak)
            precomputed_embeddings: Optional precomputed embeddings array.
                If provided, skips embedding computation. Shape must be
                (len(train_texts), embedding_dim).

        Returns:
            self
        """

    @abstractmethod
    def _predict_proba_raw(self, texts: list[str]) -> np.ndarray:
        """
        Return raw P(jailbreak) scores without validation.

        Subclasses implement this; validation happens in predict_proba().

        Args:
            texts: List of text samples

        Returns:
            Array of shape (n_samples,) with P(jailbreak) for each text
        """

    @abstractmethod
    def _predict_proba_raw_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return raw P(jailbreak) scores from precomputed embeddings.

        Subclasses implement this; validation happens in predict_proba_from_embeddings().

        Args:
            embeddings: Precomputed embeddings array of shape (n_samples, embedding_dim)

        Returns:
            Array of shape (n_samples,) with P(jailbreak) for each sample
        """

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Return P(jailbreak) for each text with degenerate score protection.

        Raises RuntimeError if scores are degenerate (std < 1e-6), indicating
        the model likely failed to train properly.

        Args:
            texts: List of text samples

        Returns:
            Array of shape (n_samples,) with P(jailbreak) for each text

        Raises:
            RuntimeError: If scores are degenerate (constant or near-constant)
        """
        scores = self._predict_proba_raw(texts)
        scores = np.asarray(scores, dtype=float)

        # Validate scores once per prediction batch
        if scores.size > 0:
            score_std = float(np.std(scores))
            if score_std < 1e-6:
                raise RuntimeError(
                    f"Scores degenerate (std={score_std:.2e} < 1e-6) — "
                    "model likely failed to train or produces constant predictions. "
                    f"Score range: [{scores.min():.4f}, {scores.max():.4f}]"
                )

        return scores

    def predict_proba_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return P(jailbreak) from precomputed embeddings with degenerate score protection.

        Raises RuntimeError if scores are degenerate (std < 1e-6), indicating
        the model likely failed to train properly.

        Args:
            embeddings: Precomputed embeddings array of shape (n_samples, embedding_dim)

        Returns:
            Array of shape (n_samples,) with P(jailbreak) for each sample

        Raises:
            RuntimeError: If scores are degenerate (constant or near-constant)
        """
        scores = self._predict_proba_raw_from_embeddings(embeddings)
        scores = np.asarray(scores, dtype=float)

        # Validate scores once per prediction batch
        if scores.size > 0:
            score_std = float(np.std(scores))
            if score_std < 1e-6:
                raise RuntimeError(
                    f"Scores degenerate (std={score_std:.2e} < 1e-6) — "
                    "model likely failed to train or produces constant predictions. "
                    f"Score range: [{scores.min():.4f}, {scores.max():.4f}]"
                )

        return scores

    def _effective_threshold(self) -> float:
        """Return calibrated threshold if available, otherwise default."""
        return self.threshold_ if self.threshold_ is not None else self.default_threshold

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Predict binary labels using threshold on P(jailbreak).

        Args:
            texts: List of text samples

        Returns:
            Array of predicted labels (0=safe, 1=jailbreak)
        """
        scores = self.predict_proba(texts)
        threshold = self._effective_threshold()
        return (scores >= threshold).astype(int)

    def calibrate_threshold(
        self,
        val_texts: list[str],
        val_labels: list[int],
        n_thresholds: int = 50,
        metric: str = "f1",
    ) -> float:
        """
        Calibrate classification threshold on validation data.

        NOTE: This method is OPTIONAL. Default behavior uses threshold=0.5.
        Call this explicitly only when threshold calibration is needed
        (e.g., for HYP-JB-004 experiments or specific use cases).

        Args:
            val_texts: Validation text samples
            val_labels: Validation labels (0=safe, 1=jailbreak)
            n_thresholds: Number of threshold candidates to try
            metric: Optimization metric. Currently only "f1" is supported.
                    Future versions may add "recall_at_fpr05", "balanced", etc.

        Returns:
            Optimal threshold value (also stored in self.threshold_)

        Raises:
            ValueError: If validation data is empty or lengths mismatch
            RuntimeError: If scores are degenerate (via predict_proba)
        """
        if not val_texts:
            raise ValueError("Validation texts are empty, cannot calibrate threshold.")

        if len(val_texts) != len(val_labels):
            raise ValueError(
                f"Length mismatch: val_texts={len(val_texts)}, val_labels={len(val_labels)}"
            )

        if metric != "f1":
            raise ValueError(
                f"Unsupported metric '{metric}'. Currently only 'f1' is supported."
            )

        y_true = np.asarray(val_labels)
        # predict_proba includes degenerate score check
        scores = self.predict_proba(val_texts)

        thresholds = np.linspace(float(scores.min()), float(scores.max()), n_thresholds)
        best_threshold = float(thresholds[0])
        best_score = -1.0

        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            score = f1_oos(y_true, y_pred, oos_label=self.positive_label)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        self.threshold_ = best_threshold
        return best_threshold
