from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .base import BaseFrameworkWrapper

LOGGER = logging.getLogger(__name__)


class H2OWrapper(BaseFrameworkWrapper):
    """
    H2O AutoML wrapper for OOS detection.

    Uses pre-trained sentence embeddings (multilingual-e5-large-instruct) as features
    and trains H2O AutoML on the resulting tabular vectors.
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        embedder_name: str = "intfloat/multilingual-e5-large-instruct",
        max_models: int = 5,
        max_runtime_secs: int = 600,
        seed: int = 42,
    ):
        super().__init__(model_name="h2o_threshold", default_threshold=default_threshold)
        self.embedder_name = embedder_name
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed

        self._embedder: SentenceTransformer | None = None
        self._aml = None
        self._label_to_id: dict[int, str] = {}
        self._id_to_label: dict[str, int] = {}
        self._feature_names: list[str] = []
        self._h2o_initialized: bool = False

    def __del__(self):
        if self._h2o_initialized:
            try:
                import h2o
                h2o.cluster().shutdown()
                LOGGER.info("H2O cluster shut down.")
            except Exception:
                pass

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

    def fit(self, train_texts: list[str], train_labels: list[int]) -> "H2OWrapper":
        try:
            import h2o
            from h2o.automl import H2OAutoML
        except ImportError as exc:
            raise ImportError("H2O is not installed. Install h2o to run this wrapper.") from exc

        x_texts = [t for t, y in zip(train_texts, train_labels) if y != self.oos_label]
        y_labels = [y for y in train_labels if y != self.oos_label]
        if not x_texts:
            raise ValueError("No in-domain samples after filtering OOS labels.")

        embeddings = self._embed(x_texts)
        self._feature_names = [f"f_{idx}" for idx in range(embeddings.shape[1])]

        unique_labels = sorted(set(y_labels))
        self._label_to_id = {label: str(i) for i, label in enumerate(unique_labels)}
        self._id_to_label = {v: k for k, v in self._label_to_id.items()}
        y_internal = [self._label_to_id[label] for label in y_labels]

        frame = pd.DataFrame(embeddings, columns=self._feature_names)
        frame["label"] = y_internal

        h2o.init()
        self._h2o_initialized = True
        train_h2o = h2o.H2OFrame(frame)
        train_h2o["label"] = train_h2o["label"].asfactor()

        aml = H2OAutoML(
            max_models=self.max_models,
            max_runtime_secs=self.max_runtime_secs,
            seed=self.seed,
            sort_metric="mean_per_class_error",
        )
        aml.train(x=self._feature_names, y="label", training_frame=train_h2o)
        self._aml = aml
        LOGGER.info(
            "H2O fit done on %d in-domain samples using %s embeddings.",
            len(x_texts),
            self.embedder_name,
        )
        return self

    def _predict_frame(self, texts: list[str]) -> pd.DataFrame:
        if self._aml is None:
            raise RuntimeError("Model is not fitted.")

        import h2o

        embeddings = self._embed(texts)
        frame = pd.DataFrame(embeddings, columns=self._feature_names)
        test_h2o = h2o.H2OFrame(frame)
        pred_h2o = self._aml.leader.predict(test_h2o)
        return pred_h2o.as_data_frame(use_pandas=True)

    def _predict_in_domain(self, texts: list[str]) -> np.ndarray:
        pred_df = self._predict_frame(texts)
        labels = pred_df["predict"].astype(str).tolist()
        return np.asarray([self._id_to_label[label] for label in labels])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        pred_df = self._predict_frame(texts)
        proba_cols = [col for col in pred_df.columns if col != "predict"]
        if not proba_cols:
            raise RuntimeError("H2O prediction does not contain class probabilities.")
        proba = pred_df[proba_cols].to_numpy(dtype=float)
        return 1.0 - proba.max(axis=1)

    def predict(self, texts: list[str]) -> np.ndarray:
        oos_scores = self.predict_proba(texts)
        in_domain_preds = self._predict_in_domain(texts)
        threshold = self._effective_threshold()
        return np.where(oos_scores >= threshold, self.oos_label, in_domain_preds)
