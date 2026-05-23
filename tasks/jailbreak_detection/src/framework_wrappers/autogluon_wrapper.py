"""
AutoGluon TabularPredictor wrapper for binary Jailbreak Detection.

Uses pre-trained sentence embeddings (multilingual-e5-large-instruct) as features
and trains AutoGluon binary classifier on the resulting vectors.
Score = P(jailbreak) directly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .base_binary import BinaryFrameworkWrapper

LOGGER = logging.getLogger(__name__)


class AutoGluonBinaryWrapper(BinaryFrameworkWrapper):
    """
    AutoGluon TabularPredictor wrapper for binary jailbreak classification.

    Key differences from OOS AutoGluonWrapper:
    - problem_type="binary" (not "multiclass")
    - fit() uses ALL samples (no OOS filtering)
    - predict_proba() returns P(jailbreak) directly (not 1-max(P))
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        embedder_name: str = "intfloat/multilingual-e5-large-instruct",
        embedder: SentenceTransformer | None = None,
        time_limit: int = 600,
        num_cpus: int = 1,
        seed: int = 42,
        verbosity: int = 0,
    ):
        """
        Initialize AutoGluon binary wrapper.

        Args:
            default_threshold: Classification threshold (default 0.5)
            embedder_name: HuggingFace model for text embeddings
            embedder: Optional pre-loaded SentenceTransformer instance.
                      If provided, embedder_name is used only for logging.
            time_limit: AutoGluon training time limit in seconds
            num_cpus: Number of CPUs for training
            seed: Random seed for reproducibility (passed to learner_kwargs)
            verbosity: AutoGluon verbosity level (0=silent, 1=info, 2+=debug)
        """
        super().__init__(
            model_name="autogluon_binary",
            default_threshold=default_threshold,
            embedder=embedder,
        )
        self.embedder_name = embedder_name
        self.time_limit = time_limit
        self.num_cpus = num_cpus
        self.seed = seed
        self.verbosity = verbosity

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
    ) -> "AutoGluonBinaryWrapper":
        """
        Fit AutoGluon binary classifier.

        Unlike OOS wrapper, trains on ALL samples (both safe and jailbreak).

        Args:
            train_texts: Training text samples
            train_labels: Training labels (0=safe, 1=jailbreak)
            precomputed_embeddings: Optional precomputed embeddings array.
                If provided, skips embedding computation. Shape must be
                (len(train_texts), embedding_dim).

        Returns:
            self

        Raises:
            ImportError: If autogluon.tabular is not installed
            ValueError: If training data is empty or embeddings shape mismatch
            RuntimeError: If AutoGluon training fails
        """
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as exc:
            raise ImportError(
                "AutoGluon is not installed. "
                "Install with: pip install 'autogluon.tabular>=1.2,<2'"
            ) from exc

        if not train_texts:
            raise ValueError("Training texts are empty.")

        n_safe = sum(1 for y in train_labels if y == 0)
        n_jailbreak = sum(1 for y in train_labels if y == 1)
        LOGGER.info(
            "Fitting AutoGluon on %d samples (safe=%d, jailbreak=%d)",
            len(train_texts),
            n_safe,
            n_jailbreak,
        )

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
        train_df = pd.DataFrame(embeddings, columns=self._feature_names)
        train_df["label"] = train_labels
        train_df["_text"] = train_texts
        train_df = train_df.sort_values(
            by=["label", "_text"],
            kind="mergesort",
        ).reset_index(drop=True)
        train_df = train_df.drop(columns=["_text"])

        # Fit AutoGluon with binary classification
        LOGGER.info(
            "Training AutoGluon (time_limit=%ds, seed=%d)...",
            self.time_limit,
            self.seed,
        )
        self._predictor = TabularPredictor(
            label="label",
            problem_type="binary",
            # FIX: Pass random_state through learner_kwargs for reproducibility
            # Without this, all runs use default seed=0 and mean±std across seeds is meaningless
            learner_kwargs={"random_state": self.seed},
        )
        self._predictor.fit(
            train_data=train_df,
            time_limit=self.time_limit,
            ag_args_fit={"num_cpus": self.num_cpus},
            verbosity=self.verbosity,
        )

        # Validate training succeeded
        lb = self._predictor.leaderboard(extra_info=False, silent=True)
        if lb is None or len(lb) == 0:
            raise RuntimeError(
                "AutoGluon produced an empty leaderboard — training likely failed. "
                "Check logs for errors."
            )

        LOGGER.info(
            "AutoGluon fit complete. Best model: %s",
            self._predictor.model_best,
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
            RuntimeError: If model not fitted or positive class missing from training
        """
        if self._predictor is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        test_df = pd.DataFrame(embeddings, columns=self._feature_names)
        proba_df = self._predictor.predict_proba(test_df)

        class_labels = list(self._predictor.class_labels)
        if self.positive_label not in class_labels:
            raise RuntimeError(
                f"Positive label {self.positive_label} not in trained "
                f"class_labels={class_labels}. Training data likely missing one of the classes."
            )
        return proba_df[self.positive_label].to_numpy()

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
