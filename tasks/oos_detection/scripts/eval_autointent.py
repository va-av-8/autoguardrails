"""
Оценка обученной модели AutoIntent на CLINC150.

Использование:
    # Оценка конкретной модели
    python scripts/eval_autointent.py --model_dir runs/autointent_classic-light_pilot_10shot_seed42

    # Оценка с указанием режима (если metadata отсутствует)
    python scripts/eval_autointent.py --model_dir runs/my_model --mode 10shot --pilot

Модель должна быть предварительно обучена с помощью train_autointent.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
script_dir = Path(__file__).parent
task_dir = script_dir.parent
project_root = task_dir.parent.parent
sys.path.insert(0, str(project_root))

from autointent import Pipeline

sys.path.insert(0, str(task_dir))
from src.data_utils import load_split
from src.metrics import compute_all_metrics, measure_latency, get_oos_scores_from_pipeline
from src.evaluation import Evaluator, EvaluationResult


def get_data_dir() -> Path:
    return task_dir / "data" / "processed"


def get_results_dir() -> Path:
    results_dir = task_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


class PipelineWrapper:
    """Wrapper for AutoIntent pipeline to match OOS model interface."""

    def __init__(self, pipe: Pipeline):
        self.pipe = pipe

    def predict(self, texts: list[str]) -> np.ndarray:
        preds = self.pipe.predict(texts)
        return np.array([-1 if p is None else p for p in preds])


def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoIntent on CLINC150")
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--source",
        choices=["standard", "deeppavlov"],
        default="deeppavlov",
        help="Data source: standard (100 OOS) or deeppavlov (200 OOS)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Mode string (e.g., '10shot', 'full'). Auto-detected from metadata if not provided.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=None,
        help="Mark as pilot run. Auto-detected from metadata if not provided.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.is_absolute():
        model_dir = task_dir / model_dir

    data_dir = get_data_dir()
    results_dir = get_results_dir()

    # Check model exists
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        print()
        print("Train a model first:")
        print("  python scripts/train_autointent.py --mode fewshot --n_shots 10 --seed 42 --pilot")
        sys.exit(1)

    # Load metadata if available
    metadata_file = model_dir / "train_metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        print(f"Loaded metadata from {metadata_file}")
    else:
        metadata = {}
        print("No metadata file found, using command-line arguments")

    # Determine parameters
    model_name = metadata.get("model_name", model_dir.name)
    source = metadata.get("source", args.source)  # Prefer metadata, fallback to args
    mode_str = args.mode or metadata.get("mode", "unknown")
    n_shots = metadata.get("n_shots")
    seed = metadata.get("seed")
    embedder_name = metadata.get("embedder", "unknown")
    pilot = args.pilot if args.pilot is not None else metadata.get("pilot", False)
    decision_metric = metadata.get("decision_metric", "decision_accuracy")

    print()
    print("=" * 60)
    print("AutoIntent Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Source: {source}")
    print(f"Mode: {mode_str}")
    print(f"Embedder: {embedder_name}")
    print(f"Decision metric: {decision_metric}")
    print(f"Pilot: {pilot}")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model from {model_dir}...")
    pipeline = Pipeline.load(model_dir)
    print("Model loaded!")
    print()

    # Load test data (use source from metadata or args)
    test_std = load_split(source, "test")
    test_texts = test_std["texts"]
    test_labels = np.array(test_std["labels"])

    print(f"Test samples: {len(test_texts)}")
    print()

    # Predictions
    print("Running predictions...")
    raw_preds = pipeline.predict(test_texts)
    y_pred = np.array([-1 if p is None else p for p in raw_preds])
    print(f"Predictions: {len(y_pred)}")
    print(f"OOS predictions: {(y_pred == -1).sum()}")

    # OOS scores for AUROC - extract continuous scores from pipeline
    y_scores = get_oos_scores_from_pipeline(pipeline, test_texts)
    n_unique = len(np.unique(y_scores))
    print(f"OOS scores: {n_unique} unique values ({'continuous' if n_unique > 2 else 'binary fallback'})")

    # Compute metrics
    print()
    print("Computing metrics...")
    metrics = compute_all_metrics(
        y_true=test_labels,
        y_scores=y_scores,
        y_pred=y_pred,
    )

    # Measure latency
    wrapper = PipelineWrapper(pipeline)
    latency = measure_latency(wrapper, test_texts[:100])

    # Save result
    result = EvaluationResult(
        model_name=model_name,
        mode=mode_str,
        oos_recall=metrics["oos_recall"],
        in_domain_acc=metrics["in_domain_acc"],
        f1_oos=metrics["f1_oos"],
        auroc=metrics["auroc"],
        au_ioc=metrics["au_ioc"],
        latency_ms=latency,
        n_shots=n_shots,
        seed=seed,
        extra={
            "source": source,
            "preset": metadata.get("preset", "classic-light"),
            "embedder": embedder_name,
            "decision_metric": decision_metric,
            "pilot": pilot,
            "comparable_to_table3": not pilot and source == "deeppavlov",
            "model_dir": str(model_dir),
        },
    )

    evaluator = Evaluator(test_std, results_dir)
    evaluator.save(result)

    # Print results
    print()
    print("=" * 60)
    print(f"AutoIntent Results ({source}, {mode_str})")
    print("=" * 60)
    print(f"  OOS Recall:    {metrics['oos_recall']:.4f}")
    print(f"  In-Domain Acc: {metrics['in_domain_acc']:.4f}")
    print(f"  F1 OOS:        {metrics['f1_oos']:.4f}")
    print(f"  AUROC:         {metrics['auroc']:.4f}")
    print(f"  AU-IOC:        {metrics['au_ioc']:.4f}")
    print(f"  Latency:       {latency:.2f} ms")

    if pilot:
        print()
        print("=" * 60)
        print("PILOT run: results NOT comparable to AutoIntent Table 3")
        print("Re-train without --pilot for final results")
        print("=" * 60)

    print()
    evaluator.print_report()


if __name__ == "__main__":
    main()
