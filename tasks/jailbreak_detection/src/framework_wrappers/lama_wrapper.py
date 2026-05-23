"""
LightAutoML (LAMA) wrapper for binary Jailbreak Detection.

Uses pre-trained sentence embeddings (multilingual-e5-large-instruct) as features
and trains LightAutoML binary classifier on the resulting vectors.
Score = P(jailbreak) directly.

IMPORTANT: Environment variables for thread control MUST be set before any
other imports to prevent hangs on Mac CPU due to multiprocessing fork/spawn
issues and OMP oversubscription.
"""

from __future__ import annotations

# LESSON FROM OOS: Set thread limits BEFORE importing numpy/torch/lightautoml
# LightAutoML on Mac CPU hangs due to multiprocessing fork/spawn and OMP oversubscription
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .base_binary import BinaryFrameworkWrapper

LOGGER = logging.getLogger(__name__)


class LAMABinaryWrapper(BinaryFrameworkWrapper):
    """
    LightAutoML (LAMA) wrapper for binary jailbreak classification.

    Key features:
    - Task("binary") for binary classification
    - fit() uses ALL samples (no OOS filtering)
    - predict_proba() returns P(jailbreak) directly
    - cpu_limit=1 to prevent Mac CPU hangs
    - Only numeric features to prevent fasttext download attempts
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        embedder: SentenceTransformer | None = None,
        embedder_name: str = "intfloat/multilingual-e5-large-instruct",
        timeout: int = 600,
        cpu_limit: int = 1,
        seed: int = 42,
    ):
        """
        Initialize LightAutoML binary wrapper.

        Args:
            default_threshold: Classification threshold (default 0.5)
            embedder: Optional pre-loaded SentenceTransformer instance.
                      If provided, embedder_name is used only for logging.
            embedder_name: HuggingFace model for text embeddings
            timeout: LightAutoML training timeout in seconds
            cpu_limit: CPU limit for training (1 to prevent Mac hangs)
            seed: Random seed for reproducibility
        """
        super().__init__(
            model_name="lama_binary",
            default_threshold=default_threshold,
            embedder=embedder,
        )
        self.embedder_name = embedder_name
        self.timeout = timeout
        self.cpu_limit = cpu_limit
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
    ) -> "LAMABinaryWrapper":
        """
        Fit LightAutoML binary classifier.

        Args:
            train_texts: Training text samples
            train_labels: Training labels (0=safe, 1=jailbreak)
            precomputed_embeddings: Optional precomputed embeddings array.
                If provided, skips embedding computation. Shape must be
                (len(train_texts), embedding_dim).

        Returns:
            self

        Raises:
            ImportError: If lightautoml is not installed
            ValueError: If training data is empty or embeddings shape mismatch
            RuntimeError: If LightAutoML training fails
        """
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
        except ImportError as exc:
            raise ImportError(
                "LightAutoML is not installed. Install lightautoml to run lama wrapper."
            ) from exc

        if not train_texts:
            raise ValueError("Training texts are empty.")

        n_safe = sum(1 for y in train_labels if y == 0)
        n_jailbreak = sum(1 for y in train_labels if y == 1)
        LOGGER.info(
            "Fitting LightAutoML on %d samples (safe=%d, jailbreak=%d)",
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
        # Only numeric features + label, NO text columns (prevent fasttext download)
        train_df = pd.DataFrame(embeddings, columns=self._feature_names)
        train_df["label"] = train_labels
        train_df["_text"] = train_texts
        train_df = train_df.sort_values(
            by=["label", "_text"],
            kind="mergesort",
        ).reset_index(drop=True)
        train_df = train_df.drop(columns=["_text"])

        # Configure LightAutoML (matching OOS etalon parameters)
        LOGGER.info(
            "Training LightAutoML (timeout=%ds, cpu_limit=%d, seed=%d)...",
            self.timeout,
            self.cpu_limit,
            self.seed,
        )
        task = Task("binary")
        automl = TabularAutoML(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.cpu_limit,
            reader_params={"random_state": self.seed},
        )

        # Train - fit_predict trains and returns predictions on train set (ignored)
        automl.fit_predict(train_df, roles={"target": "label"}, verbose=0)
        self._predictor = automl

        LOGGER.info(
            "LAMA fit done on %d samples using %s embeddings.",
            len(train_texts),
            self.embedder_name,
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

        # Build DataFrame with only numeric features (no text columns)
        test_df = pd.DataFrame(embeddings, columns=self._feature_names)

        # LightAutoML predict returns LAMLDataset with .data attribute
        preds = self._predictor.predict(test_df)
        proba = preds.data if hasattr(preds, "data") else np.asarray(preds)

        # Handle different output shapes
        # Binary: usually (n_samples, 1) with P(class=1) or (n_samples, 2) with [P(0), P(1)]
        if proba.ndim == 2:
            if proba.shape[1] == 1:
                # Shape (n, 1): squeeze to 1D
                return proba.squeeze(axis=1).astype(float)
            elif proba.shape[1] == 2:
                # Shape (n, 2): take column 1 for P(jailbreak)
                return proba[:, 1].astype(float)
            else:
                raise RuntimeError(
                    f"Unexpected LightAutoML prediction shape: {proba.shape}. "
                    "Expected (n, 1) or (n, 2) for binary classification."
                )
        elif proba.ndim == 1:
            # Already 1D
            return proba.astype(float)
        else:
            raise RuntimeError(
                f"Unexpected LightAutoML prediction shape: {proba.shape}. "
                "Expected 1D or 2D array for binary classification."
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
