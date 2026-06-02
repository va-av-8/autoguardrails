#!/usr/bin/env python3
"""
Kaggle helper: H2O argmax re-run using the same stack as run_framework_benchmarks.py.

Usage (from repo root on Kaggle):
    OOS_METRICS_LOG=compact OOS_QUIET_FIT=1 python tasks/oos_detection/scripts/kaggle_h2o_argmax_rerun.py \\
        --results-file /kaggle/working/automl_frameworks_argmax_metrics.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.oos_detection.src.experiment_runner import (  # noqa: E402
    is_degenerate_result,
    run_single_experiment,
)

LOGGER = logging.getLogger(__name__)
SOURCE = "deeppavlov"
SEEDS = [42, 123, 456]
N_SHOTS_LIST = [10, 20, 50]
EMBEDDER = "intfloat/multilingual-e5-large-instruct"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING)


def _is_data_prepared(source: str) -> bool:
    source_dir = TASK_DIR / "data" / "processed" / source
    return (source_dir / "full.json").exists() and (source_dir / "fewshot.json").exists()


def _prepare_data() -> None:
    if _is_data_prepared(SOURCE):
        return
    cmd = [sys.executable, str(SCRIPT_DIR / "prepare_data.py"), "--source", SOURCE]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def _purge_degenerate_h2o(results_file: Path) -> int:
    if not results_file.exists():
        return 0
    rows = json.loads(results_file.read_text(encoding="utf-8"))
    kept = [r for r in rows if not (r.get("model_name") == "h2o_argmax" and is_degenerate_result(r))]
    removed = len(rows) - len(kept)
    if removed:
        results_file.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    return removed


def _already_done(results_file: Path, mode_str: str, seed: int | None) -> bool:
    if not results_file.exists():
        return False
    rows = json.loads(results_file.read_text(encoding="utf-8"))
    key = ("h2o_argmax", mode_str, seed, SOURCE)
    for r in rows:
        ek = (r.get("model_name"), r.get("mode"), r.get("seed"), (r.get("extra") or {}).get("source"))
        if ek != key or is_degenerate_result(r):
            continue
        if r.get("f1_oos") is not None and r.get("in_domain_acc") is not None:
            return True
    return False


def _jobs() -> list[dict]:
    jobs: list[dict] = []
    for seed in SEEDS:
        jobs.append({"framework_name": "h2o", "source": SOURCE, "mode": "full", "n_shots": None, "seed": seed})
    for n_shots in N_SHOTS_LIST:
        for seed in SEEDS:
            jobs.append(
                {
                    "framework_name": "h2o",
                    "source": SOURCE,
                    "mode": "fewshot",
                    "n_shots": n_shots,
                    "seed": seed,
                }
            )
    return jobs


def _wrapper_kwargs(mode: str) -> dict:
    kw = {
        "embedder_name": EMBEDDER,
        "prediction_mode": "argmax",
        "max_models": 5,
    }
    kw["max_runtime_secs"] = 900 if mode == "full" else 600
    return kw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle H2O argmax re-run (repo-native).")
    parser.add_argument(
        "--results-file",
        default=str(TASK_DIR / "results" / "metrics.json"),
        help="metrics.json path (use /kaggle/working/... on Kaggle).",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    results_file = Path(args.results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    _prepare_data()
    removed = _purge_degenerate_h2o(results_file)

    jobs = _jobs()
    errors: list[dict] = []
    ok = 0

    for i, job in enumerate(jobs, 1):
        mode_str = "full" if job["mode"] == "full" else f"{job['n_shots']}shot"
        seed = job["seed"]
        meta = {
            "run": i,
            "total": len(jobs),
            "framework": "h2o",
            "mode": mode_str,
            "n_shots": job["n_shots"],
            "seed": seed,
            "prediction_mode": "argmax",
            "embedder": EMBEDDER,
            "source": SOURCE,
        }
        print("RUN_START " + json.dumps(meta, ensure_ascii=False))

        if _already_done(results_file, mode_str, seed):
            rec = {**meta, "status": "skipped", "reason": "already_in_metrics_json"}
            print("RUN_FINISH " + json.dumps(rec, ensure_ascii=False))
            continue

        try:
            run_single_experiment(
                framework_name="h2o",
                source=SOURCE,
                mode=job["mode"],
                n_shots=job["n_shots"],
                seed=seed,
                results_file=results_file,
                wrapper_kwargs={**_wrapper_kwargs(job["mode"]), "seed": seed if seed is not None else 42},
                calibrate_threshold=False,
                prediction_mode="argmax",
            )
            ok += 1
        except Exception as exc:
            err = {**meta, "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)}
            errors.append(err)
            print("RUN_FINISH " + json.dumps(err, ensure_ascii=False))
            continue

    print(
        json.dumps(
            {
                "event": "GRID_DONE",
                "n_ok": ok,
                "n_failed": len(errors),
                "purged_degenerate": removed,
                "metrics_file": str(results_file),
            },
            ensure_ascii=False,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
