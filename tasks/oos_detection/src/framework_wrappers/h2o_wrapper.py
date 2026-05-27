from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .base import BaseFrameworkWrapper

LOGGER = logging.getLogger(__name__)

# In-scope intents are 0..149; OOS (-1) is encoded as 150 for H2O (avoids factor "0" = OOS bug).
H2O_OOS_INTERNAL = 150


def _h2o_quiet_enabled() -> bool:
    for key in ("OOS_QUIET_FIT", "OOS_H2O_QUIET"):
        if os.environ.get(key, "").lower() in ("1", "true", "yes"):
            return True
    return False


@contextlib.contextmanager
def _suppress_h2o_output():
    """Hide H2O cluster banner, progress bars, and pandas conversion warnings."""
    if not _h2o_quiet_enabled():
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield


class H2OWrapper(BaseFrameworkWrapper):
    """
    H2O AutoML wrapper for OOS detection.

    Uses pre-trained sentence embeddings as features and H2O AutoML on tabular vectors.
    Labels are encoded as ints 0..149 (in-scope) and 150 (OOS) — never string factors.
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        embedder_name: str = "intfloat/multilingual-e5-large-instruct",
        max_models: int = 5,
        max_runtime_secs: int = 1500,
        seed: int = 42,
        prediction_mode: str = "threshold",
    ):
        suffix = "_argmax" if prediction_mode == "argmax" else "_threshold"
        super().__init__(
            model_name=f"h2o{suffix}",
            default_threshold=default_threshold,
            prediction_mode=prediction_mode,
        )
        self.embedder_name = embedder_name
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed

        self._embedder: SentenceTransformer | None = None
        self._aml = None
        self._feature_names: list[str] = []
        self._h2o_initialized: bool = False
        self._oos_internal = H2O_OOS_INTERNAL

    def __del__(self):
        self.release()

    @staticmethod
    def _encode_label(label: int, oos_label: int = -1) -> int:
        return H2O_OOS_INTERNAL if label == oos_label else int(label)

    def _decode_label(self, raw) -> int:
        val = int(float(raw))
        return self.oos_label if val == self._oos_internal else val

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

    def _init_h2o(self) -> None:
        import h2o

        with _suppress_h2o_output():
            try:
                h2o.cluster().shutdown(prompt=False)
            except Exception:
                pass
            h2o.init(
                strict_version_check=False,
                log_level="ERRR",
                nthreads=int(os.environ.get("H2O_NTHREADS", "2")),
                max_mem_size=os.environ.get("H2O_MAX_MEM", "6G"),
            )
        self._h2o_initialized = True

    def fit(self, train_texts: list[str], train_labels: list[int]) -> "H2OWrapper":
        try:
            import h2o
            from h2o.automl import H2OAutoML
        except ImportError as exc:
            raise ImportError("H2O is not installed. Install h2o to run this wrapper.") from exc

        x_texts, y_labels = self._train_texts_labels(train_texts, train_labels)
        y_internal = [self._encode_label(label) for label in y_labels]

        embeddings = self._embed(x_texts)
        self._feature_names = [f"f_{idx}" for idx in range(embeddings.shape[1])]

        frame = pd.DataFrame(embeddings, columns=self._feature_names)
        frame["label"] = y_internal

        self._init_h2o()
        with _suppress_h2o_output():
            train_h2o = h2o.H2OFrame(frame)
            train_h2o["label"] = train_h2o["label"].asfactor()

            aml = H2OAutoML(
                max_models=self.max_models,
                max_runtime_secs=self.max_runtime_secs,
                seed=self.seed,
                sort_metric="logloss",
                balance_classes=False,
            )
            aml.train(x=self._feature_names, y="label", training_frame=train_h2o)
        if aml.leader is None:
            raise RuntimeError(
                "H2O AutoML produced no leader model (empty or failed run). "
                "Try increasing max_runtime_secs or check the training frame."
            )
        self._aml = aml
        LOGGER.info("H2O fit done on %d samples.", len(x_texts))
        return self

    def release(self) -> None:
        """Shut down H2O cluster to free JVM heap between Kaggle runs."""
        if not self._h2o_initialized:
            return
        try:
            import h2o

            with _suppress_h2o_output():
                h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass
        self._h2o_initialized = False
        self._aml = None

    def _predict_frame(self, texts: list[str]) -> pd.DataFrame:
        if self._aml is None:
            raise RuntimeError("Model is not fitted.")

        import h2o

        embeddings = self._embed(texts)
        frame = pd.DataFrame(embeddings, columns=self._feature_names)
        with _suppress_h2o_output():
            test_h2o = h2o.H2OFrame(frame)
            pred_h2o = self._aml.leader.predict(test_h2o)
            return pred_h2o.as_data_frame(use_pandas=True)

    def _predict_in_domain(self, texts: list[str]) -> np.ndarray:
        pred_df = self._predict_frame(texts)
        return np.asarray([self._decode_label(x) for x in pred_df["predict"]])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        pred_df = self._predict_frame(texts)
        oos_col = f"p{self._oos_internal}"
        if oos_col in pred_df.columns:
            return pred_df[oos_col].to_numpy(dtype=float)
        proba_cols = [c for c in pred_df.columns if c != "predict"]
        if not proba_cols:
            raise RuntimeError("H2O prediction does not contain class probabilities.")
        proba = pred_df[proba_cols].to_numpy(dtype=float)
        return 1.0 - proba.max(axis=1)

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.prediction_mode == "argmax":
            return self._predict_in_domain(texts)
        oos_scores = self.predict_proba(texts)
        in_domain_preds = self._predict_in_domain(texts)
        threshold = self._effective_threshold()
        return np.where(oos_scores >= threshold, self.oos_label, in_domain_preds)
