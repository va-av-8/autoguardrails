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
        timeout: int = 3600,
        cpu_limit: int = 1,
        seed: int = 42,
        prediction_mode: str = "threshold",
    ):
        suffix = "_argmax" if prediction_mode == "argmax" else "_threshold"
        super().__init__(
            model_name=f"lama{suffix}",
            default_threshold=default_threshold,
            prediction_mode=prediction_mode,
        )
        self.embedder_name = embedder_name
        self.timeout = timeout
        self.cpu_limit = cpu_limit
        self.seed = seed

        self._embedder: SentenceTransformer | None = None
        self._automl = None
        self._class_labels: list[int] = []
        self._feature_names: list[str] = []
        # Embedding cache: (tuple of texts) -> embeddings
        self._embed_cache: tuple[tuple[str, ...], np.ndarray] | None = None

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        key = tuple(texts)
        if self._embed_cache is not None and self._embed_cache[0] == key:
            return self._embed_cache[1]
        model = self._get_embedder()
        embeddings = np.asarray(
            model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        self._embed_cache = (key, embeddings)
        return embeddings

    def fit(self, train_texts: list[str], train_labels: list[int]) -> "LAMAWrapper":
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
        except ImportError as exc:
            raise ImportError(
                "LightAutoML is not installed. Install lightautoml to run lama wrapper."
            ) from exc

        x_texts, y_labels = self._train_texts_labels(train_texts, train_labels)

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
        if self.prediction_mode == "argmax" and self.oos_label in self._class_labels:
            oos_idx = self._class_labels.index(self.oos_label)
            return proba[:, oos_idx]
        return 1.0 - proba.max(axis=1)

    def predict_proba_full(self, texts: list[str]) -> np.ndarray:
        """Return full probability matrix [n_samples, n_classes]."""
        return self._predict_proba_matrix(texts)

    def get_classes(self) -> list[int]:
        """Return list of class labels in column order of predict_proba_full."""
        return list(self._class_labels)

    def get_model_info(self) -> dict | None:
        """Return LAMA model info (algorithms used, blender weights if available)."""
        if self._automl is None:
            return None
        info: dict = {"framework": "lama"}
        try:
            # Try to extract used models from reader/pipes
            if hasattr(self._automl, "reader") and self._automl.reader is not None:
                info["reader_class"] = type(self._automl.reader).__name__
            if hasattr(self._automl, "blender") and self._automl.blender is not None:
                blender = self._automl.blender
                info["blender_class"] = type(blender).__name__
                if hasattr(blender, "wts"):
                    info["blender_weights"] = list(blender.wts) if blender.wts is not None else None
            # Extract pipe names
            if hasattr(self._automl, "levels") and self._automl.levels:
                pipes = []
                for level in self._automl.levels:
                    for pipe in level:
                        pipes.append(type(pipe).__name__)
                info["pipes"] = pipes
        except Exception:
            info["extraction_error"] = "Could not extract full model info"
        return info

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.prediction_mode == "argmax":
            return self._predict_in_domain(texts)
        oos_scores = self.predict_proba(texts)
        in_domain_preds = self._predict_in_domain(texts)
        threshold = self._effective_threshold()
        return np.where(oos_scores >= threshold, self.oos_label, in_domain_preds)
