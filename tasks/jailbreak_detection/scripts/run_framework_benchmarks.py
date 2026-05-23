#!/usr/bin/env python3
"""
CLI script to run Jailbreak Detection AutoML benchmark grid.

Usage examples:
    # Run pilot: AutoGluon × 10shot × seed=42
    uv run python tasks/jailbreak_detection/scripts/run_framework_benchmarks.py \
        --frameworks autogluon --n-shots 10 --seeds 42 --run-fewshot

    # Run full grid: all frameworks × all modes × all seeds
    uv run python tasks/jailbreak_detection/scripts/run_framework_benchmarks.py \
        --run-full --run-fewshot

    # Run H2O and LAMA on full-train only
    uv run python tasks/jailbreak_detection/scripts/run_framework_benchmarks.py \
        --frameworks h2o lama --run-full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path.
SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from tasks.jailbreak_detection.src.experiment_runner import run_grid

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDER = "intfloat/multilingual-e5-large-instruct"
DEFAULT_RESULTS_FILE = "tasks/jailbreak_detection/results/metrics.json"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Jailbreak Detection AutoML benchmark grid."
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["autogluon", "h2o", "lama"],
        choices=["autogluon", "h2o", "lama"],
        help="Framework wrappers to run (default: all).",
    )
    parser.add_argument(
        "--n-shots",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        dest="n_shots",
        help="Few-shot setup values (default: 10 20 50).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Few-shot random seeds (default: 42 123 456).",
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="Run full-train experiments.",
    )
    parser.add_argument(
        "--run-fewshot",
        action="store_true",
        help="Run few-shot experiments.",
    )
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to metrics.json output file (default: {DEFAULT_RESULTS_FILE}).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue grid run if one experiment fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose results are already present in metrics.json.",
    )
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        help=f"HuggingFace embedder model (default: {DEFAULT_EMBEDDER}).",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()

    run_full = args.run_full
    run_fewshot = args.run_fewshot
    if not run_full and not run_fewshot:
        LOGGER.info("No mode flags provided, enabling both --run-full and --run-fewshot.")
        run_full = True
        run_fewshot = True

    # Load embedder once
    LOGGER.info("Loading embedder: %s", args.embedder)
    embedder = SentenceTransformer(args.embedder)

    results, errors, skipped_count = run_grid(
        frameworks=args.frameworks,
        run_full=run_full,
        run_fewshot=run_fewshot,
        n_shots_list=args.n_shots,
        seeds=args.seeds,
        embedder=embedder,
        embedder_hf_model=args.embedder,
        results_file=Path(args.results_file),
        continue_on_error=args.continue_on_error,
        skip_existing=args.skip_existing,
    )

    LOGGER.info(
        "Grid finished: %d completed, %d skipped, %d failed",
        len(results),
        skipped_count,
        len(errors),
    )
    if errors:
        for err in errors:
            LOGGER.warning(
                "FAILED framework=%s mode=%s n_shots=%s seed=%s error=%s",
                err["framework"],
                err["mode"],
                err["n_shots"],
                err["seed"],
                err["error"],
            )
        if not args.continue_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
