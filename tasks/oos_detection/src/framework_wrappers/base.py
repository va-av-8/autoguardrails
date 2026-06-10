from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from tasks.oos_detection.src.metrics import f1_oos

PredictionMode = Literal["argmax", "threshold"]


class BaseFrameworkWrapper(ABC):
    """Common interface for framework-based OOS wrappers."""

    oos_label: int = -1

    def __init__(
        self,
        model_name: str,
        default_threshold: float = 0.5,
        prediction_mode: PredictionMode = "threshold",
    ):
        if prediction_mode not in ("argmax", "threshold"):
            raise ValueError(f"prediction_mode must be 'argmax' or 'threshold', got {prediction_mode!r}")
        self.prediction_mode: PredictionMode = prediction_mode
        self.model_name = model_name
        self.default_threshold = default_threshold
        self.threshold_: float | None = None

    def _train_texts_labels(
        self,
        train_texts: list[str],
        train_labels: list[int],
    ) -> tuple[list[str], list[int]]:
        """Argmax: OOS as a regular class; threshold: in-scope only."""
        if self.prediction_mode == "argmax":
            return list(train_texts), list(train_labels)
        x_texts = [t for t, y in zip(train_texts, train_labels) if y != self.oos_label]
        y_labels = [y for y in train_labels if y != self.oos_label]
        if not x_texts:
            raise ValueError("No in-domain samples after filtering OOS labels.")
        return x_texts, y_labels

    @abstractmethod
    def fit(self, train_texts: list[str], train_labels: list[int]) -> "BaseFrameworkWrapper":
        """Fit wrapper on training data."""

    @abstractmethod
    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict final labels with OOS handling."""

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return OOS score for each text (higher = more OOS)."""

    @abstractmethod
    def predict_proba_full(self, texts: list[str]) -> np.ndarray:
        """Return full probability matrix [n_samples, n_classes]."""

    @abstractmethod
    def get_classes(self) -> list[int]:
        """Return list of class labels in column order of predict_proba_full."""

    @abstractmethod
    def _predict_in_domain(self, texts: list[str]) -> np.ndarray:
        """Predict in-domain labels only."""

    def _effective_threshold(self) -> float:
        return self.threshold_ if self.threshold_ is not None else self.default_threshold

    def calibrate_threshold(
        self,
        val_texts: list[str],
        val_labels: list[int],
        n_thresholds: int = 50,
    ) -> float:
        """
        Calibrate OOS threshold on validation split by maximizing F1 OOS.
        """
        if not val_texts:
            raise ValueError("Validation texts are empty, cannot calibrate threshold.")

        y_true = np.asarray(val_labels)
        oos_scores = np.asarray(self.predict_proba(val_texts), dtype=float)
        y_in_domain = np.asarray(self._predict_in_domain(val_texts))

        if oos_scores.size == 0:
            raise ValueError("predict_proba returned no OOS scores for validation texts.")

        score_std = float(np.std(oos_scores))
        if score_std < 1e-6:
            raise RuntimeError(
                "OOS scores degenerate (std < 1e-6) — model likely failed to train or scores are constant."
            )

        thresholds = np.linspace(float(oos_scores.min()), float(oos_scores.max()), n_thresholds)
        best_threshold = float(thresholds[0])
        best_f1 = -1.0

        for threshold in thresholds:
            y_pred = np.where(oos_scores >= threshold, self.oos_label, y_in_domain)
            score = f1_oos(y_true, y_pred, oos_label=self.oos_label)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        self.threshold_ = best_threshold
        return best_threshold
