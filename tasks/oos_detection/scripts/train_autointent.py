"""
Обучение AutoIntent на CLINC150.

Использование:
    # Pilot (маленький embedder, быстрая валидация)
    python scripts/train_autointent.py --mode fewshot --n_shots 10 --seed 42 --pilot

    # Final (оригинальный embedder, сравнимо с Table 3)
    python scripts/train_autointent.py --mode fewshot --n_shots 10 --seed 42

    # Full train
    python scripts/train_autointent.py --mode full

После обучения модель сохраняется в runs/<model_name>/.
Для оценки используйте eval_autointent.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load .env for macOS ARM fix (OMP_NUM_THREADS=1, etc.)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add project root to path
script_dir = Path(__file__).parent
task_dir = script_dir.parent
project_root = task_dir.parent.parent
sys.path.insert(0, str(project_root))

from autointent import Pipeline, Dataset as AIDataset
from autointent.configs import LoggingConfig, EmbedderConfig, DataConfig

sys.path.insert(0, str(task_dir / "src"))
from data_utils import (
    load_split_autointent,
    load_fewshot_autointent,
    get_intents,
)


def get_data_dir() -> Path:
    return task_dir / "data" / "processed"


def get_runs_dir() -> Path:
    runs_dir = task_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_model_name(pilot: bool) -> str:
    if pilot:
        return "autointent_classic-light_pilot"
    return "autointent_classic-light"


def get_embedder_name(pilot: bool) -> str:
    if pilot:
        return "intfloat/multilingual-e5-small"
    return "intfloat/multilingual-e5-large-instruct"


def main():
    parser = argparse.ArgumentParser(description="Train AutoIntent on CLINC150")
    parser.add_argument(
        "--source",
        choices=["standard", "deeppavlov"],
        default="deeppavlov",
        help="Data source: standard (100 OOS) or deeppavlov (200 OOS)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "fewshot"],
        default="fewshot",
        help="Training mode",
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
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use small embedder for fast validation (not comparable to Table 3)",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    runs_dir = get_runs_dir()

    # Model naming
    model_name = get_model_name(args.pilot)
    embedder_name = get_embedder_name(args.pilot)

    if args.mode == "fewshot":
        mode_str = f"{args.n_shots}shot"
    else:
        mode_str = "full"

    # Output directory for this run
    model_dir = runs_dir / f"{model_name}_{mode_str}_seed{args.seed if args.seed else 0}"

    print("=" * 60)
    print("AutoIntent Training")
    print("=" * 60)
    print(f"Mode: {'PILOT' if args.pilot else 'FINAL'}")
    print(f"Embedder: {embedder_name}")
    print(f"Training: {mode_str}")
    print(f"Output: {model_dir}")
    print("=" * 60)
    print()

    # Load data (AutoIntent format)
    if args.mode == "fewshot":
        train_ai = load_fewshot_autointent(args.source, args.n_shots, args.seed)
    else:
        train_ai = load_split_autointent(args.source, "train")

    test_ai = load_split_autointent(args.source, "test")
    intents = get_intents(args.source)

    print(f"Train: {len(train_ai)} samples")
    print(f"Test: {len(test_ai)} samples")
    print(f"Intents: {len(intents)}")
    print()

    # Create AutoIntent Dataset
    ai_dataset = AIDataset.from_dict({
        "train": train_ai,
        "test": test_ai,
        "intents": intents,
    })

    # Create pipeline
    pipeline = Pipeline.from_preset("classic-light")
    pipeline.set_config(EmbedderConfig(model_name=embedder_name))

    # Cross-validation for few-shot
    if args.mode == "fewshot":
        pipeline.set_config(DataConfig(scheme="cv", n_folds=3))

    # Logging config - save modules for later loading
    pipeline.set_config(LoggingConfig(
        project_dir=model_dir,
        dump_modules=True,  # Important: save for eval_autointent.py
        clear_ram=False,    # Keep in RAM for immediate dump
    ))

    # Train
    print(f"Starting AutoML optimization...")
    print(f"This may take a while (20 trials with cross-validation)")
    print()

    context = pipeline.fit(ai_dataset)
    print()
    print("AutoML optimization completed!")

    # Save model
    print(f"Saving model to {model_dir}...")
    pipeline.dump(model_dir)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "mode": mode_str,
        "n_shots": args.n_shots if args.mode == "fewshot" else None,
        "seed": args.seed if args.mode == "fewshot" else None,
        "embedder": embedder_name,
        "pilot": args.pilot,
        "preset": "classic-light",
    }

    import json
    metadata_file = model_dir / "train_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    print()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Model saved to: {model_dir}")
    print()
    print("Next step - evaluate:")
    print(f"  python scripts/eval_autointent.py --model_dir {model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
