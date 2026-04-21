"""
Запуск AutoIntent на CLINC150 (обучение + оценка).

Это обёртка для последовательного вызова:
    1. train_autointent.py — обучение и сохранение модели
    2. eval_autointent.py — загрузка и оценка

Использование:
    # Pilot (маленький embedder, быстрая валидация)
    python scripts/run_autointent.py --source standard --mode fewshot --n_shots 10 --seed 42 --pilot

    # Final (e5-large-instruct, сравнимо с Table 3)
    python scripts/run_autointent.py --source deeppavlov --mode fewshot --n_shots 10 --seed 42

    # Full train
    python scripts/run_autointent.py --source standard --mode full

    # AutoML оптимизация embedder
    python scripts/run_autointent.py --source standard --mode fewshot --n_shots 10 --seed 42 --no-fix-embedder

Или запустите скрипты отдельно:
    python scripts/train_autointent.py --source standard --mode fewshot --n_shots 10 --seed 42
    python scripts/eval_autointent.py --model_dir runs/autointent_classic-light_standard_10shot_seed42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
task_dir = script_dir.parent


def get_model_dir(pilot: bool, no_fix_embedder: bool, source: str, mode: str, n_shots: int | None, seed: int | None) -> Path:
    """Construct model directory path."""
    if no_fix_embedder:
        model_name = "autointent_classic-light_autoembedder"
    elif pilot:
        model_name = "autointent_classic-light_pilot"
    else:
        model_name = "autointent_classic-light"

    if mode == "fewshot":
        mode_str = f"{n_shots}shot"
    else:
        mode_str = "full"

    seed_val = seed if seed else 0
    return task_dir / "runs" / f"{model_name}_{source}_{mode_str}_seed{seed_val}"


def main():
    parser = argparse.ArgumentParser(
        description="Run AutoIntent on CLINC150 (train + evaluate)"
    )
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
        help="Use small embedder for fast validation",
    )
    parser.add_argument(
        "--no-fix-embedder",
        action="store_true",
        help="Let AutoML optimize embedder (slower, potentially better)",
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

    # Build command arguments
    no_fix_embedder = getattr(args, 'no_fix_embedder', False)

    train_args = [
        sys.executable,
        str(script_dir / "train_autointent.py"),
        "--source", args.source,
        "--mode", args.mode,
        "--n_shots", str(args.n_shots),
        "--seed", str(args.seed),
    ]
    if args.pilot:
        train_args.append("--pilot")
    if no_fix_embedder:
        train_args.append("--no-fix-embedder")

    model_dir = get_model_dir(
        args.pilot, no_fix_embedder, args.source, args.mode,
        args.n_shots if args.mode == "fewshot" else None,
        args.seed,  # always pass seed for consistent model_dir
    )

    eval_args = [
        sys.executable,
        str(script_dir / "eval_autointent.py"),
        "--model_dir", str(model_dir),
    ]

    # Run training
    if not args.eval_only:
        print("=" * 60)
        print("STEP 1: Training")
        print("=" * 60)
        result = subprocess.run(train_args)
        if result.returncode != 0:
            print("Training failed!")
            sys.exit(result.returncode)
        print()

    # Run evaluation
    if not args.train_only:
        print("=" * 60)
        print("STEP 2: Evaluation")
        print("=" * 60)
        result = subprocess.run(eval_args)
        if result.returncode != 0:
            print("Evaluation failed!")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
