#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Add project root to path.
SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.oos_detection.src.experiment_runner import run_framework_grid

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _is_data_prepared(source: str) -> bool:
    source_dir = TASK_DIR / "data" / "processed" / source
    return (source_dir / "full.json").exists() and (source_dir / "fewshot.json").exists()


def _prepare_data_if_needed(sources: list[str], prepare_if_missing: bool) -> None:
    missing = [source for source in sources if not _is_data_prepared(source)]
    if not missing:
        return

    if not prepare_if_missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            "Prepared data is missing for sources: "
            f"{missing_str}. Run: uv run python tasks/oos_detection/scripts/prepare_data.py --source all"
        )

    for source in missing:
        LOGGER.info("Prepared data is missing for '%s', running prepare_data.py", source)
        cmd = [
            sys.executable,
            str(TASK_DIR / "scripts" / "prepare_data.py"),
            "--source",
            source,
        ]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOS framework benchmark grid.")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["autogluon", "h2o", "lama"],
        choices=["autogluon", "h2o", "lama"],
        help="Framework wrappers to run.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["standard", "deeppavlov"],
        choices=["standard", "deeppavlov"],
        help="Prepared data sources.",
    )
    parser.add_argument(
        "--n-shots",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        dest="n_shots",
        help="Few-shot setup values.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Few-shot random seeds.",
    )
    parser.add_argument("--run-full", action="store_true", help="Run full-train experiments.")
    parser.add_argument("--run-fewshot", action="store_true", help="Run few-shot experiments.")
    parser.add_argument(
        "--results-file",
        default="tasks/oos_detection/results/metrics.json",
        help="Path to metrics.json output file.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue grid run if one experiment fails.",
    )
    parser.add_argument(
        "--prepare-if-missing",
        action="store_true",
        help="Run existing prepare_data.py script when processed data is missing.",
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

    _prepare_data_if_needed(args.sources, args.prepare_if_missing)

    results, errors = run_framework_grid(
        frameworks=args.frameworks,
        sources=args.sources,
        run_full=run_full,
        run_fewshot=run_fewshot,
        n_shots_list=args.n_shots,
        seeds=args.seeds,
        results_file=args.results_file,
        continue_on_error=args.continue_on_error,
    )

    LOGGER.info("Completed experiments: %d", len(results))
    if errors:
        LOGGER.warning("Failed experiments: %d", len(errors))
        for err in errors:
            LOGGER.warning(
                "FAILED framework=%s source=%s mode=%s n_shots=%s seed=%s error=%s",
                err["framework"],
                err["source"],
                err["mode"],
                err["n_shots"],
                err["seed"],
                err["error"],
            )
        if not args.continue_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
