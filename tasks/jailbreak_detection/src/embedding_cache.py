"""
Embedding cache utilities for Jailbreak Detection AutoML experiments.

Caches precomputed embeddings to avoid recomputing them across experiments.
Cache is stored in data/processed/embeddings_cache/ as .npy files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)

CACHE_DIR = Path("tasks/jailbreak_detection/data/processed/embeddings_cache")


def _ensure_cache_dir() -> None:
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def safe_embedder_name(embedder_hf_model: str) -> str:
    """
    Convert HuggingFace model name to safe filename component.

    Args:
        embedder_hf_model: HF model name like "intfloat/multilingual-e5-large-instruct"

    Returns:
        Safe string like "intfloat_multilingual-e5-large-instruct"
    """
    return embedder_hf_model.replace("/", "_")


def cache_path(embedder_hf_model: str, split_id: str) -> Path:
    """
    Get path to cached embeddings file.

    Args:
        embedder_hf_model: HF model name
        split_id: Split identifier like "train_shot10_seed42" or "test"

    Returns:
        Path to .npy cache file
    """
    _ensure_cache_dir()
    safe_name = safe_embedder_name(embedder_hf_model)
    return CACHE_DIR / f"{safe_name}_{split_id}.npy"


def get_or_compute_embeddings(
    embedder: "SentenceTransformer",
    embedder_hf_model: str,
    split_id: str,
    texts: list[str],
) -> np.ndarray:
    """
    Get embeddings from cache or compute and cache them.

    Args:
        embedder: SentenceTransformer instance
        embedder_hf_model: HF model name (for cache path)
        split_id: Split identifier like "train_shot10_seed42" or "test"
        texts: List of texts to embed

    Returns:
        Embeddings array of shape (len(texts), embedding_dim)
    """
    path = cache_path(embedder_hf_model, split_id)

    if path.exists():
        LOGGER.info("cache hit: %s", path)
        embeddings = np.load(path)
        # Validate cached embeddings match expected size
        if embeddings.shape[0] != len(texts):
            LOGGER.warning(
                "Cached embeddings size mismatch: cached=%d, expected=%d. Recomputing.",
                embeddings.shape[0],
                len(texts),
            )
        else:
            return embeddings

    # Cache miss — compute embeddings
    LOGGER.info("cache miss, computing embeddings for %s, n=%d", split_id, len(texts))
    embeddings = np.asarray(
        embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    # Save to cache
    _ensure_cache_dir()
    np.save(path, embeddings)
    LOGGER.info("Saved embeddings to cache: %s", path)

    return embeddings
