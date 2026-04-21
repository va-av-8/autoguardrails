from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseFrameworkWrapper

LOGGER = logging.getLogger(__name__)


class H2OWrapper(BaseFrameworkWrapper):
    """
    H2O AutoML wrapper for OOS detection.

    H2O backend does not provide a robust native text pipeline for this setup,
    so we use TF-IDF features and train H2O AutoML on tabular vectors.
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        max_models: int = 5,
        max_runtime_secs: int = 600,
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        seed: int = 42,
    ):
        super().__init__(model_name="h2o_threshold", default_threshold=default_threshold)
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self._aml = None
        self._label_to_id: dict[int, str] = {}
        self._id_to_label: dict[str, int] = {}
        self._feature_names: list[str] = []

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

        x_matrix = self._vectorizer.fit_transform(x_texts)
        x_dense = x_matrix.toarray()
        self._feature_names = [f"f_{idx}" for idx in range(x_dense.shape[1])]

        unique_labels = sorted(set(y_labels))
        self._label_to_id = {label: str(i) for i, label in enumerate(unique_labels)}
        self._id_to_label = {v: k for k, v in self._label_to_id.items()}
        y_internal = [self._label_to_id[label] for label in y_labels]

        frame = pd.DataFrame(x_dense, columns=self._feature_names)
        frame["label"] = y_internal

        h2o.init()
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
            "H2O fit done on %d in-domain samples. e5-large-instruct external embedder is not supported for this backend.",
            len(x_texts),
        )
        return self

    def _predict_frame(self, texts: list[str]) -> pd.DataFrame:
        if self._aml is None:
            raise RuntimeError("Model is not fitted.")

        import h2o

        x_matrix = self._vectorizer.transform(texts).toarray()
        frame = pd.DataFrame(x_matrix, columns=self._feature_names)
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
