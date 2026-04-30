from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .base import BaseFrameworkWrapper

LOGGER = logging.getLogger(__name__)


class AutoGluonWrapper(BaseFrameworkWrapper):
    """
    AutoGluon TabularPredictor wrapper for OOS detection.

    Uses pre-trained sentence embeddings (multilingual-e5-large-instruct) as features
    and trains AutoGluon TabularPredictor on the resulting tabular vectors.
    OOS score = 1 - max_class_probability.
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        embedder_name: str = "intfloat/multilingual-e5-large-instruct",
        time_limit: int | None = 600,
        num_cpus: int = 1,
        seed: int = 42,
    ):
        super().__init__(model_name="autogluon_threshold", default_threshold=default_threshold)
        self.embedder_name = embedder_name
        self.time_limit = time_limit
        self.num_cpus = num_cpus
        self.seed = seed

        self._embedder: SentenceTransformer | None = None
        self._predictor = None
        self._classes: np.ndarray | None = None
        self._feature_names: list[str] = []

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_embedder()
        return np.asarray(
            model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def fit(self, train_texts: list[str], train_labels: list[int]) -> "AutoGluonWrapper":
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as exc:
            raise ImportError(
                "AutoGluon is not installed. Install autogluon.tabular to run this wrapper."
            ) from exc

        x_texts = [t for t, y in zip(train_texts, train_labels) if y != self.oos_label]
        y_labels = [y for y in train_labels if y != self.oos_label]
        if not x_texts:
            raise ValueError("No in-domain samples after filtering OOS labels.")

        embeddings = self._embed(x_texts)
        self._feature_names = [f"f_{idx}" for idx in range(embeddings.shape[1])]
        self._classes = np.array(sorted(set(y_labels)))

        train_df = pd.DataFrame(embeddings, columns=self._feature_names)
        train_df["label"] = y_labels

        self._predictor = TabularPredictor(
            label="label",
            problem_type="multiclass",
            learner_kwargs={"random_state": self.seed},
        )
        self._predictor.fit(
            train_data=train_df,
            time_limit=self.time_limit,
            ag_args_fit={"num_cpus": self.num_cpus},
            verbosity=0,
        )
        lb = self._predictor.leaderboard(extra_info=False, silent=True)
        if lb is None or len(lb) == 0:
            raise RuntimeError(
                "AutoGluon produced an empty leaderboard — training likely failed."
            )

        LOGGER.info(
            "AutoGluon fit done on %d in-domain samples using %s embeddings.",
            len(x_texts),
            self.embedder_name,
        )
        return self

    def _predict_in_domain(self, texts: list[str]) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError("Model is not fitted.")
        embeddings = self._embed(texts)
        test_df = pd.DataFrame(embeddings, columns=self._feature_names)
        preds = self._predictor.predict(test_df)
        return np.asarray(preds.tolist() if hasattr(preds, "tolist") else list(preds))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError("Model is not fitted.")
        embeddings = self._embed(texts)
        test_df = pd.DataFrame(embeddings, columns=self._feature_names)
        proba_df = self._predictor.predict_proba(test_df)
        proba = proba_df.to_numpy()
        return 1.0 - proba.max(axis=1)

    def predict(self, texts: list[str]) -> np.ndarray:
        oos_scores = self.predict_proba(texts)
        in_domain_preds = self._predict_in_domain(texts)
        threshold = self._effective_threshold()
        return np.where(oos_scores >= threshold, self.oos_label, in_domain_preds)
