from __future__ import annotations

import logging
import os

# Mac CPU settings to avoid multiprocessing fork/spawn issues and OMP oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .base import BaseFrameworkWrapper

LOGGER = logging.getLogger(__name__)


class LAMAWrapper(BaseFrameworkWrapper):
    """
    LightAutoML (LAMA) wrapper for OOS detection.

    LAMA is treated as LightAutoML tabular classification over text embeddings.
    We use `intfloat/multilingual-e5-large-instruct` for text representation.
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        embedder_name: str = "intfloat/multilingual-e5-large-instruct",
        timeout: int = 600,
        cpu_limit: int = 1,
        seed: int = 42,
    ):
        super().__init__(model_name="lama_threshold", default_threshold=default_threshold)
        self.embedder_name = embedder_name
        self.timeout = timeout
        self.cpu_limit = cpu_limit
        self.seed = seed

        self._embedder: SentenceTransformer | None = None
        self._automl = None
        self._class_labels: list[int] = []
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

    def fit(self, train_texts: list[str], train_labels: list[int]) -> "LAMAWrapper":
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
        except ImportError as exc:
            raise ImportError(
                "LightAutoML is not installed. Install lightautoml to run lama wrapper."
            ) from exc

        x_texts = [t for t, y in zip(train_texts, train_labels) if y != self.oos_label]
        y_labels = [y for y in train_labels if y != self.oos_label]
        if not x_texts:
            raise ValueError("No in-domain samples after filtering OOS labels.")

        embeddings = self._embed(x_texts)
        self._feature_names = [f"f_{idx}" for idx in range(embeddings.shape[1])]
        self._class_labels = sorted(set(y_labels))
        label_to_index = {label: idx for idx, label in enumerate(self._class_labels)}
        y_internal = [label_to_index[label] for label in y_labels]

        train_df = pd.DataFrame(embeddings, columns=self._feature_names)
        train_df["label"] = y_internal
        train_df["_text"] = x_texts
        train_df = train_df.sort_values(by=["label", "_text"], kind="mergesort").reset_index(drop=True)
        train_df = train_df.drop(columns=["_text"])

        task = Task("multiclass")
        automl = TabularAutoML(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.cpu_limit,
            reader_params={"random_state": self.seed},
        )
        automl.fit_predict(train_df, roles={"target": "label"}, verbose=0)
        self._automl = automl
        LOGGER.info("LAMA fit done on %d in-domain samples using %s.", len(x_texts), self.embedder_name)
        return self

    def _predict_proba_matrix(self, texts: list[str]) -> np.ndarray:
        if self._automl is None:
            raise RuntimeError("Model is not fitted.")
        embeddings = self._embed(texts)
        test_df = pd.DataFrame(embeddings, columns=self._feature_names)
        pred = self._automl.predict(test_df)
        proba = pred.data if hasattr(pred, "data") else pred
        return np.asarray(proba)

    def _predict_in_domain(self, texts: list[str]) -> np.ndarray:
        proba = self._predict_proba_matrix(texts)
        idx = proba.argmax(axis=1)
        return np.asarray([self._class_labels[i] for i in idx])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = self._predict_proba_matrix(texts)
        return 1.0 - proba.max(axis=1)

    def predict(self, texts: list[str]) -> np.ndarray:
        oos_scores = self.predict_proba(texts)
        in_domain_preds = self._predict_in_domain(texts)
        threshold = self._effective_threshold()
        return np.where(oos_scores >= threshold, self.oos_label, in_domain_preds)
