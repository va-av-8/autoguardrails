"""
Запуск AutoIntent на WildJailbreak (обучение + оценка).

Использование:
    # Pilot (маленький embedder, быстрая валидация)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --pilot

    # Final (оригинал��ный embedder)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42

    # Только оценка (модель должна существовать)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --eval-only

Данные:
    - Few-shot train: data/processed/train_shot{N}_seed{S}.json
    - Test: data/processed/test.json (для inference)
    - Eval binary: data/processed/wildjailbreak_eval_binary.jsonl (для метрик)

Результаты:
    - Модель: runs/autointent_classic-light_{N}shot_seed{S}/
    - Метрики: results/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Load .env for macOS ARM fix (OMP_NUM_THREADS=1, etc.)
from dotenv import load_dotenv

# Add project root to path
script_dir = Path(__file__).parent
task_dir = script_dir.parent
project_root = task_dir.parent.parent
load_dotenv(project_root / ".env")

sys.path.insert(0, str(project_root))

from autointent import Pipeline, Dataset as AIDataset
from autointent.configs import LoggingConfig, EmbedderConfig, DataConfig

from tasks.jailbreak_detection.src.metrics import evaluate_jailbreak


def get_data_dir() -> Path:
    return task_dir / "data" / "processed"


def get_runs_dir() -> Path:
    runs_dir = task_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_results_dir() -> Path:
    results_dir = task_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_model_name(pilot: bool) -> str:
    if pilot:
        return "autointent_classic-light_pilot"
    return "autointent_classic-light"


def get_embedder_name(pilot: bool) -> str:
    if pilot:
        return "intfloat/multilingual-e5-small"
    return "intfloat/multilingual-e5-large-instruct"


def get_model_dir(pilot: bool, n_shots: int, seed: int) -> Path:
    model_name = get_model_name(pilot)
    return get_runs_dir() / f"{model_name}_{n_shots}shot_seed{seed}"


def load_fewshot_train(n_shots: int, seed: int, data_dir: Path) -> dict:
    """
    Load few-shot train data in AutoIntent format.
    Format: {"intents": [{"id": 0, "name": "safe", "utterances": [...]}],
             "oos_utterances": [...]}
    """
    path = data_dir / f"train_shot{n_shots}_seed{seed}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test(data_dir: Path) -> dict:
    """
    Load test data.
    Format: {"utterances": [...], "labels": [...]}
    Labels: 0 = safe, 1 = jailbreak
    """
    path = data_dir / "test.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_binary(data_dir: Path) -> list[dict]:
    """
    Load eval binary data for metrics computation.
    Contains: prompt, binary_label, data_type
    """
    path = data_dir / "wildjailbreak_eval_binary.jsonl"
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def convert_to_autointent_test(test_data: dict) -> list[dict]:
    """
    Convert test data to AutoIntent format for inference.
    Binary classification (no OOS):
    - safe (label=0): {"utterance": str, "label": 0}
    - jailbreak (label=1): {"utterance": str, "label": 1}
    """
    result = []
    for utt, label in zip(test_data["utterances"], test_data["labels"]):
        result.append({"utterance": utt, "label": label})
    return result


def convert_to_autointent_train(train_data: dict) -> list[dict]:
    """
    Convert few-shot train data to AutoIntent list format.
    Binary classification:
    - safe (from intents[0]) -> label 0
    - jailbreak (from oos_utterances) -> label 1
    """
    result = []
    # Safe examples (label=0)
    for intent in train_data["intents"]:
        for utt in intent["utterances"]:
            result.append({"utterance": utt, "label": 0})
    # Jailbreak examples (label=1) - NOT OOS, but second intent
    for utt in train_data["oos_utterances"]:
        result.append({"utterance": utt, "label": 1})
    return result


def save_metrics(result: dict, results_dir: Path) -> None:
    """Save metrics to metrics.json, appending to existing list."""
    metrics_path = results_dir / "metrics.json"

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        if not isinstance(all_results, list):
            # Convert dict format to list if needed
            all_results = list(all_results.values()) if all_results else []
    else:
        all_results = []

    # Remove existing entry with same model_name, mode, n_shots, seed
    all_results = [
        r for r in all_results
        if not (
            r.get("model_name") == result["model_name"] and
            r.get("mode") == result["mode"] and
            r.get("n_shots") == result.get("n_shots") and
            r.get("seed") == result.get("seed")
        )
    ]

    all_results.append(result)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved to {metrics_path}")


def train(args, data_dir: Path, model_dir: Path) -> None:
    """Train AutoIntent pipeline."""
    embedder_name = get_embedder_name(args.pilot)

    print("=" * 60)
    print("AutoIntent Training (Jailbreak Detection)")
    print("=" * 60)
    print(f"Mode: {'PILOT' if args.pilot else 'FINAL'}")
    print(f"Embedder: {embedder_name}")
    print(f"Training: {args.n_shots}shot, seed={args.seed}")
    print(f"Output: {model_dir}")
    print("=" * 60)
    print()

    # Load data
    train_raw = load_fewshot_train(args.n_shots, args.seed, data_dir)
    test_raw = load_test(data_dir)

    train_ai = convert_to_autointent_train(train_raw)
    test_ai = convert_to_autointent_test(test_raw)

    # Binary classification: 2 intents (no OOS)
    intents = [
        {"id": 0, "name": "safe"},
        {"id": 1, "name": "jailbreak"},
    ]

    n_safe = len(train_raw['intents'][0]['utterances'])
    n_jailbreak = len(train_raw['oos_utterances'])
    print(f"Train: {len(train_ai)} samples")
    print(f"  - safe (label=0): {n_safe}")
    print(f"  - jailbreak (label=1): {n_jailbreak}")
    print(f"Test: {len(test_ai)} samples")
    print()

    # Create AutoIntent Dataset
    ai_dataset = AIDataset.from_dict({
        "train": train_ai,
        "test": test_ai,
        "intents": intents,
    })

    # Create pipeline with classic-light preset
    pipeline = Pipeline.from_preset("classic-light")
    pipeline.set_config(EmbedderConfig(model_name=embedder_name))

    # Cross-validation for few-shot
    pipeline.set_config(DataConfig(scheme="cv", n_folds=3))

    # Logging config
    pipeline.set_config(LoggingConfig(
        project_dir=model_dir,
        dump_modules=True,
        clear_ram=False,
    ))

    # Train
    print("Starting AutoML optimization...")
    print("This may take a while...")
    print()

    pipeline.fit(ai_dataset)
    print()
    print("AutoML optimization completed!")

    # Save model
    print(f"Saving model to {model_dir}...")
    pipeline.dump(model_dir)

    # Save metadata
    metadata = {
        "model_name": get_model_name(args.pilot),
        "mode": f"{args.n_shots}shot",
        "n_shots": args.n_shots,
        "seed": args.seed,
        "embedder": embedder_name,
        "pilot": args.pilot,
        "preset": "classic-light",
        "task": "jailbreak_detection",
        "approach": "binary_classification",  # Not OOS detection
    }
    (model_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2))

    print()
    print("Training completed!")


def evaluate(args, data_dir: Path, model_dir: Path, results_dir: Path) -> None:
    """Evaluate trained AutoIntent pipeline."""
    print()
    print("=" * 60)
    print("AutoIntent Evaluation (Jailbreak Detection)")
    print("=" * 60)

    # Load metadata
    metadata_path = model_dir / "train_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {
            "model_name": get_model_name(args.pilot),
            "mode": f"{args.n_shots}shot",
            "n_shots": args.n_shots,
            "seed": args.seed,
            "embedder": get_embedder_name(args.pilot),
            "pilot": args.pilot,
        }

    print(f"Model: {metadata['model_name']}")
    print(f"Mode: {metadata['mode']}")
    print(f"Embedder: {metadata['embedder']}")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model from {model_dir}...")
    pipeline = Pipeline.load(model_dir)
    print("Model loaded!")
    print()

    # Load test data
    test_raw = load_test(data_dir)
    test_texts = test_raw["utterances"]

    # Load eval binary for data_types and ground truth
    eval_binary = load_eval_binary(data_dir)
    y_true = np.array([1 if r["binary_label"] == "jailbreak" else 0 for r in eval_binary])
    data_types = np.array([r["data_type"] for r in eval_binary])

    print(f"Test samples: {len(test_texts)}")
    print(f"  safe: {sum(y_true == 0)}, jailbreak: {sum(y_true == 1)}")
    print()

    # Predictions
    print("Running predictions...")
    raw_preds = pipeline.predict(test_texts)

    # Binary classification: predictions are 0 (safe) or 1 (jailbreak) directly
    # None means model couldn't decide - treat as jailbreak (safer for guardrail)
    y_pred = np.array([1 if p is None else p for p in raw_preds])

    print(f"Predictions: {len(y_pred)}")
    print(f"  predicted safe: {sum(y_pred == 0)}")
    print(f"  predicted jailbreak: {sum(y_pred == 1)}")
    print()

    # Compute metrics
    print("Computing metrics...")
    metrics = evaluate_jailbreak(y_true, y_pred, data_types, oos_label=1)

    # Print results
    print()
    print("=" * 60)
    print(f"Results ({metadata['mode']})")
    print("=" * 60)
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    # Save results
    result = {
        "model_name": metadata["model_name"],
        "mode": metadata["mode"],
        "n_shots": metadata["n_shots"],
        "seed": metadata["seed"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "over_refusal_rate": metrics["over_refusal_rate"],
        "recall_adversarial_harmful": metrics.get("recall_adversarial_harmful"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extra": {
            "preset": metadata.get("preset", "classic-light"),
            "embedder": metadata["embedder"],
            "pilot": metadata["pilot"],
            "model_dir": str(model_dir),
        },
    }

    save_metrics(result, results_dir)

    if metadata.get("pilot"):
        print()
        print("=" * 60)
        print("PILOT run: using small embedder for fast validation")
        print("Re-run without --pilot for final results")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run AutoIntent on WildJailbreak (train + evaluate)"
    )
    parser.add_argument(
        "--mode",
        choices=["fewshot"],
        default="fewshot",
        help="Training mode (only fewshot supported for jailbreak)",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        choices=[10, 20, 50],
        default=10,
        help="Number of shots per class",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=[42, 123, 456],
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use small embedder for fast validation",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train, skip evaluation",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate (model must exist)",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    results_dir = get_results_dir()
    model_dir = get_model_dir(args.pilot, args.n_shots, args.seed)

    # Train
    if not args.eval_only:
        train(args, data_dir, model_dir)

    # Evaluate
    if not args.train_only:
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            print("Train a model first (remove --eval-only flag)")
            sys.exit(1)
        evaluate(args, data_dir, model_dir, results_dir)


if __name__ == "__main__":
    main()
