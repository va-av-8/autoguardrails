"""
Запуск бейзлайнов: TF-IDF + LogReg и Cosine Similarity.

Использование:
    # Все бейзлайны (full train)
    python scripts/run_baseline.py --model all

    # Только TF-IDF
    python scripts/run_baseline.py --model tfidf

    # Только cosine
    python scripts/run_baseline.py --model cosine

    # Конкретная embedding-модель
    python scripts/run_baseline.py --model cosine --embedding_model bert-base-uncased

    # Few-shot режим
    python scripts/run_baseline.py --model all --mode fewshot --n_shots 10 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
task_dir = script_dir.parent
project_root = task_dir.parent.parent
sys.path.insert(0, str(project_root))

from baselines.tfidf_logreg import TfidfLogreg
from baselines.embedding_threshold import EmbeddingThreshold, SUPPORTED_MODELS
from shared.data_utils import load_clinc150, load_fewshot
from shared.evaluation import Evaluator


def get_data_dir() -> Path:
    return task_dir / "data" / "processed"


def get_cache_dir() -> Path:
    return get_data_dir() / "embeddings_cache"


def get_results_dir() -> Path:
    results_dir = task_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_tfidf(
    train_data: dict,
    val_data: dict,
    evaluator: Evaluator,
    mode_str: str,
    n_shots: int | None,
    seed: int | None,
) -> None:
    """Run TF-IDF baselines: argmax and threshold variants."""

    # Argmax variant (no threshold calibration)
    print(f"Running tfidf_argmax ({mode_str})...")
    model_argmax = TfidfLogreg()
    model_argmax.fit(train_data["texts"], train_data["labels"])
    result_argmax = evaluator.evaluate(
        model=model_argmax,
        model_name="tfidf_argmax",
        mode=mode_str,
        n_shots=n_shots,
        seed=seed,
    )
    evaluator.save(result_argmax)

    # Threshold variant (calibrated on validation)
    print(f"Running tfidf_threshold ({mode_str})...")
    model_thresh = TfidfLogreg()
    model_thresh.fit(train_data["texts"], train_data["labels"])
    t_star = model_thresh.calibrate_threshold(val_data["texts"], val_data["labels"])
    print(f"  Calibrated threshold: {t_star:.4f}")
    result_thresh = evaluator.evaluate(
        model=model_thresh,
        model_name="tfidf_threshold",
        mode=mode_str,
        n_shots=n_shots,
        seed=seed,
    )
    evaluator.save(result_thresh)


def run_cosine(
    train_data: dict,
    val_data: dict,
    evaluator: Evaluator,
    mode_str: str,
    n_shots: int | None,
    seed: int | None,
    embedding_model: str | None = None,
) -> None:
    """Run cosine similarity baselines: argmax and threshold variants."""

    models_to_run = [embedding_model] if embedding_model else SUPPORTED_MODELS
    cache_dir = get_cache_dir()

    for emb_model in models_to_run:
        short_name = "bert" if "bert-base" in emb_model else "minilm"

        # Argmax variant (default threshold)
        print(f"Running cosine_{short_name}_argmax ({mode_str})...")
        model_argmax = EmbeddingThreshold(model_name=emb_model, cache_dir=cache_dir)
        model_argmax.fit(train_data["texts"], train_data["labels"])
        result_argmax = evaluator.evaluate(
            model=model_argmax,
            model_name=f"cosine_{short_name}_argmax",
            mode=mode_str,
            n_shots=n_shots,
            seed=seed,
        )
        evaluator.save(result_argmax)

        # Threshold variant (calibrated on validation)
        print(f"Running cosine_{short_name}_threshold ({mode_str})...")
        model_thresh = EmbeddingThreshold(model_name=emb_model, cache_dir=cache_dir)
        model_thresh.fit(train_data["texts"], train_data["labels"])
        t_star = model_thresh.calibrate_threshold(val_data["texts"], val_data["labels"])
        print(f"  Calibrated threshold: {t_star:.4f}")
        result_thresh = evaluator.evaluate(
            model=model_thresh,
            model_name=f"cosine_{short_name}_threshold",
            mode=mode_str,
            n_shots=n_shots,
            seed=seed,
        )
        evaluator.save(result_thresh)


def main():
    parser = argparse.ArgumentParser(description="Run OOS detection baselines")
    parser.add_argument(
        "--model",
        choices=["tfidf", "cosine", "all"],
        default="all",
        help="Which baseline to run",
    )
    parser.add_argument(
        "--embedding_model",
        choices=SUPPORTED_MODELS,
        default=None,
        help="Specific embedding model for cosine baseline",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "fewshot"],
        default="full",
        help="Training mode: full train or few-shot",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        choices=[10, 20, 50],
        default=10,
        help="Number of shots per intent (for fewshot mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=[42, 123, 456],
        default=42,
        help="Random seed (for fewshot mode)",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    results_dir = get_results_dir()

    # Load train data (depends on mode)
    if args.mode == "fewshot":
        train_data = load_fewshot(args.n_shots, args.seed, data_dir)
        mode_str = f"{args.n_shots}shot"
        n_shots = args.n_shots
        seed = args.seed
    else:
        train_data = load_clinc150("train", data_dir)
        mode_str = "full"
        n_shots = None
        seed = None

    # Val and test are ALWAYS full
    val_data = load_clinc150("validation", data_dir)
    test_data = load_clinc150("test", data_dir)

    print(f"Train: {len(train_data['texts'])} samples")
    print(f"Val: {len(val_data['texts'])} samples")
    print(f"Test: {len(test_data['texts'])} samples")
    print()

    # Create evaluator
    evaluator = Evaluator(test_data, results_dir)

    # Run baselines
    if args.model in ["tfidf", "all"]:
        run_tfidf(train_data, val_data, evaluator, mode_str, n_shots, seed)

    if args.model in ["cosine", "all"]:
        run_cosine(
            train_data, val_data, evaluator, mode_str,
            n_shots, seed, args.embedding_model
        )

    # Print report
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    evaluator.print_report()


if __name__ == "__main__":
    main()
