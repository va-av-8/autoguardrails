"""
Experiment runner for Jailbreak Detection AutoML experiments.

Orchestrates: load data → embed → fit wrapper → predict → evaluate → save metrics.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .data_utils import (
    compute_split_id,
    load_eval_data_types,
    load_test,
    load_train,
)
from .embedding_cache import get_or_compute_embeddings
from .framework_wrappers import create_wrapper
from .metrics import evaluate_jailbreak

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)

DEFAULT_RESULTS_FILE = Path("tasks/jailbreak_detection/results/metrics.json")


def append_to_metrics_json(record: dict, results_file: Path) -> None:
    """
    Append experiment record to metrics.json.

    Args:
        record: Experiment record dict
        results_file: Path to metrics.json
    """
    results_file.parent.mkdir(parents=True, exist_ok=True)

    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            all_records = json.load(f)
        if not isinstance(all_records, list):
            # Convert dict format to list if needed
            all_records = list(all_records.values()) if all_records else []
    else:
        all_records = []

    all_records.append(record)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    LOGGER.info("Appended record to %s", results_file)


def run_single(
    framework: str,
    mode: str,
    seed: int,
    n_shots: int | None,
    embedder: "SentenceTransformer",
    embedder_hf_model: str,
    results_file: Path = DEFAULT_RESULTS_FILE,
    default_threshold: float = 0.5,
    wrapper_kwargs: dict | None = None,
) -> dict:
    """
    Run a single AutoML experiment.

    Args:
        framework: Framework name ("autogluon", "h2o", "lama")
        mode: Training mode ("10shot", "20shot", "50shot", "full")
        seed: Random seed
        n_shots: Number of shots (for few-shot modes)
        embedder: Pre-loaded SentenceTransformer instance
        embedder_hf_model: HuggingFace model name for logging
        results_file: Path to metrics.json
        default_threshold: Classification threshold (default 0.5)
        wrapper_kwargs: Additional framework-specific kwargs (e.g., time_limit for AutoGluon)

    Returns:
        Experiment record dict (also appended to metrics.json)
    """
    LOGGER.info(
        "Running experiment: framework=%s mode=%s seed=%d",
        framework,
        mode,
        seed,
    )

    # 1. Load data
    train_texts, train_labels = load_train(mode, seed, n_shots)
    test_texts, test_labels = load_test()
    data_types = load_eval_data_types()

    LOGGER.info(
        "Loaded data: train=%d (safe=%d, jailbreak=%d), test=%d",
        len(train_texts),
        sum(1 for y in train_labels if y == 0),
        sum(1 for y in train_labels if y == 1),
        len(test_texts),
    )

    # 2. Get or compute embeddings
    train_split_id = compute_split_id(mode, seed, n_shots)
    train_embeddings = get_or_compute_embeddings(
        embedder, embedder_hf_model, train_split_id, train_texts
    )
    test_embeddings = get_or_compute_embeddings(
        embedder, embedder_hf_model, "test", test_texts
    )

    # 3. Create wrapper
    wrapper = create_wrapper(
        framework,
        embedder=embedder,
        embedder_name=embedder_hf_model,
        seed=seed,
        default_threshold=default_threshold,
        **(wrapper_kwargs or {}),
    )

    # 4. Fit with precomputed embeddings
    t0 = time.time()
    wrapper.fit(train_texts, train_labels, precomputed_embeddings=train_embeddings)
    training_time_sec = time.time() - t0
    LOGGER.info("Training completed in %.1f seconds", training_time_sec)

    # 5. Predict
    scores = wrapper.predict_proba_from_embeddings(test_embeddings)
    threshold = wrapper._effective_threshold()
    preds = (scores >= threshold).astype(int)

    # 6. Evaluate
    test_labels_arr = np.asarray(test_labels)
    metrics = evaluate_jailbreak(test_labels_arr, preds, data_types, oos_label=1)

    LOGGER.info(
        "Metrics: f1=%.4f precision=%.4f recall=%.4f over_refusal_rate=%.4f",
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        metrics["over_refusal_rate"],
    )

    # 7. Compute eval_counts
    TP = int(((preds == 1) & (test_labels_arr == 1)).sum())
    FP = int(((preds == 1) & (test_labels_arr == 0)).sum())
    FN = int(((preds == 0) & (test_labels_arr == 1)).sum())
    TN = int(((preds == 0) & (test_labels_arr == 0)).sum())

    eval_counts = {
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "tn": TN,
        "fnr_jailbreak": FN / max(FN + TP, 1),
        "fpr_safe": FP / max(FP + TN, 1),
        "n_eval": int(len(test_labels)),
        "n_safe_true": int((test_labels_arr == 0).sum()),
        "n_jailbreak_true": int((test_labels_arr == 1).sum()),
    }

    # 8. Compute scores_eval_summary
    # margin = 2*score - 1, compatible with AutoIntent (range [-1, 1])
    margin = 2 * scores - 1
    scores_eval_summary = {
        "n_scored": int(len(scores)),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
        "margin_min": float(np.min(margin)),
        "margin_max": float(np.max(margin)),
        "scores_mean": float(np.mean(scores)),
        "scores_std": float(np.std(scores)),
        "scores_min": float(np.min(scores)),
        "scores_max": float(np.max(scores)),
        "note": f"P(jailbreak) from {framework}, margin = 2*P - 1",
    }

    # 9. Build record (compatible with AutoIntent format)
    record = {
        "model_name": framework,
        "mode": mode,
        "n_shots": n_shots,
        "seed": seed,
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "over_refusal_rate": float(metrics["over_refusal_rate"]),
        "recall_adversarial_harmful": float(metrics.get("recall_adversarial_harmful", 0.0)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extra": {
            "embedder": "fixed (e5-large-instruct)",
            "embedder_hf_model": embedder_hf_model,
            "embedder_fixed": True,
            "eval_counts": eval_counts,
            "scores_eval_summary": scores_eval_summary,
            "scores": scores.tolist(),
            "threshold_used": float(threshold),
            "train_size": int(len(train_texts)),
            "n_safe": int(sum(1 for y in train_labels if y == 0)),
            "n_jailbreak": int(sum(1 for y in train_labels if y == 1)),
            "training_time_sec": float(training_time_sec),
        },
    }

    # 10. Save
    append_to_metrics_json(record, results_file)

    return record


def _load_existing_keys(results_file: Path) -> set[tuple]:
    """
    Load existing experiment keys from metrics.json.

    Returns:
        Set of (model_name, mode, n_shots, seed) tuples for completed experiments.
    """
    if not results_file.exists():
        return set()

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            records = list(records.values()) if records else []
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to load existing records from %s: %s", results_file, exc)
        return set()

    existing = set()
    for r in records:
        key = (
            r.get("model_name"),
            r.get("mode"),
            r.get("n_shots"),
            r.get("seed"),
        )
        existing.add(key)

    return existing


def run_grid(
    frameworks: list[str],
    run_full: bool,
    run_fewshot: bool,
    n_shots_list: list[int],
    seeds: list[int],
    embedder: "SentenceTransformer",
    embedder_hf_model: str,
    results_file: Path = DEFAULT_RESULTS_FILE,
    default_threshold: float = 0.5,
    wrapper_kwargs: dict | None = None,
    continue_on_error: bool = False,
    skip_existing: bool = False,
) -> tuple[list[dict], list[dict], int]:
    """
    Run grid of AutoML experiments.

    Args:
        frameworks: List of framework names ("autogluon", "h2o", "lama")
        run_full: Run full-train experiments
        run_fewshot: Run few-shot experiments
        n_shots_list: Few-shot setup values (e.g., [10, 20, 50])
        seeds: Random seeds for few-shot sampling
        embedder: Pre-loaded SentenceTransformer instance
        embedder_hf_model: HuggingFace model name for logging
        results_file: Path to metrics.json
        default_threshold: Classification threshold (default 0.5)
        wrapper_kwargs: Additional framework-specific kwargs
        continue_on_error: Continue grid run if one experiment fails
        skip_existing: Skip experiments already present in metrics.json

    Returns:
        Tuple of (results, errors, skipped_count) where:
        - results: List of successful experiment records
        - errors: List of error dicts with experiment metadata and error message
        - skipped_count: Number of experiments skipped due to skip_existing
    """
    if not run_full and not run_fewshot:
        raise ValueError("At least one mode must be enabled: run_full or run_fewshot")

    results: list[dict] = []
    errors: list[dict] = []
    skipped_count = 0

    # Load existing experiment keys if skip_existing is enabled
    existing_keys: set[tuple] = set()
    if skip_existing:
        existing_keys = _load_existing_keys(results_file)
        LOGGER.info("Found %d existing records in %s", len(existing_keys), results_file)

    # Build job list
    jobs: list[dict] = []
    for framework in frameworks:
        if run_full:
            for seed in seeds:
                jobs.append({
                    "framework": framework,
                    "mode": "full",
                    "n_shots": None,
                    "seed": seed,
                })
        if run_fewshot:
            for n_shots in n_shots_list:
                for seed in seeds:
                    jobs.append({
                        "framework": framework,
                        "mode": f"{n_shots}shot",
                        "n_shots": n_shots,
                        "seed": seed,
                    })

    # Run experiments
    for job in jobs:
        # Check if experiment already exists (skip_existing logic)
        # Key matches record format: (model_name=framework, mode, n_shots, seed)
        job_key = (job["framework"], job["mode"], job["n_shots"], job["seed"])
        if skip_existing and job_key in existing_keys:
            LOGGER.info(
                "SKIP: %s %s n_shots=%s seed=%s (already in metrics.json)",
                job["framework"],
                job["mode"],
                job["n_shots"],
                job["seed"],
            )
            skipped_count += 1
            continue

        try:
            LOGGER.info(
                "Running: framework=%s mode=%s n_shots=%s seed=%s",
                job["framework"],
                job["mode"],
                job["n_shots"],
                job["seed"],
            )
            record = run_single(
                framework=job["framework"],
                mode=job["mode"],
                seed=job["seed"],
                n_shots=job["n_shots"],
                embedder=embedder,
                embedder_hf_model=embedder_hf_model,
                results_file=results_file,
                default_threshold=default_threshold,
                wrapper_kwargs=wrapper_kwargs,
            )
            results.append(record)
        except Exception as exc:
            error_payload = {
                "framework": job["framework"],
                "mode": job["mode"],
                "n_shots": job["n_shots"],
                "seed": job["seed"],
                "error": str(exc),
            }
            errors.append(error_payload)
            LOGGER.exception("Experiment failed: %s", error_payload)
            if not continue_on_error:
                raise

    LOGGER.info(
        "Grid summary: %d completed, %d skipped, %d failed",
        len(results),
        skipped_count,
        len(errors),
    )

    return results, errors, skipped_count
