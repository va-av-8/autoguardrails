"""
H2O AutoML wrapper for binary Jailbreak Detection.

Uses pre-trained sentence embeddings (multilingual-e5-large-instruct) as features
and trains H2O AutoML binary classifier on the resulting vectors.
Score = P(jailbreak) directly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .base_binary import BinaryFrameworkWrapper

LOGGER = logging.getLogger(__name__)


class H2OBinaryWrapper(BinaryFrameworkWrapper):
    """
    H2O AutoML wrapper for binary jailbreak classification.

    Key features:
    - Binary classification via asfactor() on target column
    - fit() uses ALL samples (no OOS filtering)
    - predict_proba() returns P(jailbreak) directly
    - Cluster shutdown/init between runs for clean state
    - Budget controlled by max_models (not time-based), matching OOS etalon
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        embedder: SentenceTransformer | None = None,
        embedder_name: str = "intfloat/multilingual-e5-large-instruct",
        max_models: int = 5,
        seed: int = 42,
    ):
        """
        Initialize H2O binary wrapper.

        Args:
            default_threshold: Classification threshold (default 0.5)
            embedder: Optional pre-loaded SentenceTransformer instance.
                      If provided, embedder_name is used only for logging.
            embedder_name: HuggingFace model for text embeddings
            max_models: Maximum number of models to train (budget control)
            seed: Random seed for reproducibility
        """
        super().__init__(
            model_name="h2o_binary",
            default_threshold=default_threshold,
            embedder=embedder,
        )
        self.embedder_name = embedder_name
        self.max_models = max_models
        self.seed = seed

        self._predictor = None
        self._feature_names: list[str] = []

    def _get_embedder(self) -> SentenceTransformer:
        """Get sentence transformer embedder (use provided or lazy-load)."""
        if self.embedder is not None:
            return self.embedder
        # Lazy-load if not provided
        if not hasattr(self, "_lazy_embedder") or self._lazy_embedder is None:
            LOGGER.info("Loading embedder: %s", self.embedder_name)
            self._lazy_embedder = SentenceTransformer(self.embedder_name)
        return self._lazy_embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using sentence transformer."""
        model = self._get_embedder()
        return np.asarray(
            model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        precomputed_embeddings: np.ndarray | None = None,
    ) -> "H2OBinaryWrapper":
        """
        Fit H2O AutoML binary classifier.

        Args:
            train_texts: Training text samples
            train_labels: Training labels (0=safe, 1=jailbreak)
            precomputed_embeddings: Optional precomputed embeddings array.
                If provided, skips embedding computation. Shape must be
                (len(train_texts), embedding_dim).

        Returns:
            self

        Raises:
            ImportError: If h2o is not installed
            ValueError: If training data is empty or embeddings shape mismatch
            RuntimeError: If H2O training fails
        """
        try:
            import h2o
            from h2o.automl import H2OAutoML
        except ImportError as exc:
            raise ImportError(
                "H2O is not installed. Install h2o to run this wrapper."
            ) from exc

        if not train_texts:
            raise ValueError("Training texts are empty.")

        n_safe = sum(1 for y in train_labels if y == 0)
        n_jailbreak = sum(1 for y in train_labels if y == 1)
        LOGGER.info(
            "Fitting H2O on %d samples (safe=%d, jailbreak=%d)",
            len(train_texts),
            n_safe,
            n_jailbreak,
        )

        # Shutdown existing cluster before init for clean state (OOS lesson)
        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass

        h2o.init()
        self._h2o_initialized = True

        # Use precomputed embeddings or compute them
        if precomputed_embeddings is not None:
            if precomputed_embeddings.shape[0] != len(train_texts):
                raise ValueError(
                    f"Embeddings shape mismatch: embeddings have {precomputed_embeddings.shape[0]} rows, "
                    f"but train_texts has {len(train_texts)} samples"
                )
            LOGGER.info("Using precomputed embeddings (shape=%s)", precomputed_embeddings.shape)
            embeddings = precomputed_embeddings
        else:
            LOGGER.info("Embedding %d texts with %s...", len(train_texts), self.embedder_name)
            embeddings = self._embed(train_texts)
        self._feature_names = [f"f_{idx}" for idx in range(embeddings.shape[1])]

        # Create DataFrame with deterministic ordering for reproducibility
        # Only numeric features + label, NO text columns (prevent H2O Word2Vec)
        train_df = pd.DataFrame(embeddings, columns=self._feature_names)
        train_df["label"] = train_labels
        train_df["_text"] = train_texts
        train_df = train_df.sort_values(
            by=["label", "_text"],
            kind="mergesort",
        ).reset_index(drop=True)
        train_df = train_df.drop(columns=["_text"])

        # Convert to H2OFrame
        train_h2o = h2o.H2OFrame(train_df)

        # CRITICAL: Convert target to categorical for binary classification
        # Otherwise H2O treats it as regression
        train_h2o["label"] = train_h2o["label"].asfactor()

        # Train H2O AutoML (matching OOS etalon: max_models, seed, sort_metric)
        LOGGER.info(
            "Training H2O AutoML (max_models=%d, seed=%d)...",
            self.max_models,
            self.seed,
        )
        aml = H2OAutoML(
            max_models=self.max_models,
            seed=self.seed,
            sort_metric="AUC",  # AUC for binary (mean_per_class_error for multiclass in OOS)
            exclude_algos=["StackedEnsemble"],  # Avoid AssertionError on few-shot data
        )
        aml.train(
            x=self._feature_names,
            y="label",
            training_frame=train_h2o,
        )

        # Validate training succeeded
        if aml.leader is None:
            raise RuntimeError(
                "H2O AutoML produced no leader model (empty or failed run). "
                "Try increasing max_models or check the training frame."
            )

        self._predictor = aml.leader
        LOGGER.info(
            "H2O fit done on %d samples using %s embeddings. Leader: %s",
            len(train_texts),
            self.embedder_name,
            self._predictor.model_id,
        )

        return self

    def _predict_proba_raw_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return raw P(jailbreak) scores from precomputed embeddings.

        Args:
            embeddings: Precomputed embeddings array of shape (n_samples, embedding_dim)

        Returns:
            Array of P(jailbreak) for each sample

        Raises:
            RuntimeError: If model not fitted or prediction format unexpected
        """
        if self._predictor is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        import h2o

        # Build DataFrame with only numeric features (no text columns)
        test_df = pd.DataFrame(embeddings, columns=self._feature_names)
        test_h2o = h2o.H2OFrame(test_df)

        # H2O predict returns DataFrame with columns: predict, p0, p1
        preds_h2o = self._predictor.predict(test_h2o)
        preds_df = preds_h2o.as_data_frame(use_pandas=True)

        # Get P(jailbreak) = P(class=1)
        if "p1" in preds_df.columns:
            return preds_df["p1"].to_numpy().astype(float)
        else:
            raise RuntimeError(
                f"Expected 'p1' column in H2O predictions, got columns: {list(preds_df.columns)}. "
                "Binary classification may have failed."
            )

    def _predict_proba_raw(self, texts: list[str]) -> np.ndarray:
        """
        Return raw P(jailbreak) scores.

        Embeds texts and delegates to _predict_proba_raw_from_embeddings.

        Args:
            texts: List of text samples

        Returns:
            Array of P(jailbreak) for each text
        """
        embeddings = self._embed(texts)
        return self._predict_proba_raw_from_embeddings(embeddings)
