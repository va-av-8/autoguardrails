from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import BaseFrameworkWrapper

LOGGER = logging.getLogger(__name__)


class AutoGluonWrapper(BaseFrameworkWrapper):
    """
    AutoGluon-based wrapper for OOS detection.

    AutoGluon does not natively support OOS detection, so the wrapper trains
    a multiclass in-domain classifier and uses:
        OOS score = 1 - max_class_probability
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        time_limit: int | None = None,
        seed: int = 42,
    ):
        super().__init__(model_name="autogluon_threshold", default_threshold=default_threshold)
        self.time_limit = time_limit
        self.seed = seed

        self._predictor = None
        self._classes: np.ndarray | None = None

    def fit(self, train_texts: list[str], train_labels: list[int]) -> "AutoGluonWrapper":
        try:
            from autogluon.multimodal import MultiModalPredictor
        except ImportError as exc:
            raise ImportError(
                "AutoGluon is not installed. Install it to run autogluon wrapper."
            ) from exc

        train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
        train_df = train_df[train_df["label"] != self.oos_label].copy()
        if train_df.empty:
            raise ValueError("No in-domain samples after filtering OOS labels.")

        # Keep labels in original format (int) for external API compatibility.
        self._predictor = MultiModalPredictor(label="label", problem_type="multiclass")
        self._predictor.fit(
            train_data=train_df,
            seed=self.seed,
            time_limit=self.time_limit,
        )

        self._classes = np.array(sorted(train_df["label"].unique().tolist()))
        LOGGER.info("AutoGluon fit done on %d in-domain samples.", len(train_df))
        return self

    def _predict_in_domain(self, texts: list[str]) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError("Model is not fitted.")
        test_df = pd.DataFrame({"text": texts})
        preds = self._predictor.predict(test_df)
        return np.asarray(preds.tolist() if hasattr(preds, "tolist") else list(preds))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError("Model is not fitted.")
        test_df = pd.DataFrame({"text": texts})
        proba_df = self._predictor.predict_proba(test_df)
        proba = proba_df.to_numpy()
        return 1.0 - proba.max(axis=1)

    def predict(self, texts: list[str]) -> np.ndarray:
        oos_scores = self.predict_proba(texts)
        in_domain_preds = self._predict_in_domain(texts)
        threshold = self._effective_threshold()
        return np.where(oos_scores >= threshold, self.oos_label, in_domain_preds)
