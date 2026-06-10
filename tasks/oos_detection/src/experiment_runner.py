from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .data_utils import load_fewshot, load_split
from .evaluation import EvaluationResult, Evaluator
from .framework_wrappers import create_wrapper

LOGGER = logging.getLogger(__name__)
OOS_TASK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_FILE = OOS_TASK_DIR / "results" / "metrics.json"


def build_wrapper(framework_name: str, **kwargs: Any):
    """Build framework wrapper by name."""
    return create_wrapper(framework_name, **kwargs)


def load_experiment_data(
    source: str,
    mode: str,
    n_shots: int | None = None,
    seed: int | None = None,
) -> tuple[dict, dict, dict]:
    """
    Load train/validation/test data for one experiment.

    Validation and test are always taken from full split.
    """
    if mode == "fewshot":
        if n_shots is None or seed is None:
            raise ValueError("fewshot mode requires n_shots and seed")
        train_data = load_fewshot(source, n_shots=n_shots, seed=seed)
    elif mode == "full":
        train_data = load_split(source, split="train")
    else:
        raise ValueError("mode must be either 'full' or 'fewshot'")

    val_data = load_split(source, split="validation")
    test_data = load_split(source, split="test")
    return train_data, val_data, test_data


def _resolve_results_file(results_file: str | Path | None) -> Path:
    path = Path(results_file) if results_file is not None else DEFAULT_RESULTS_FILE
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_single_experiment(
    framework_name: str,
    source: str,
    mode: str,
    n_shots: int | None = None,
    seed: int | None = None,
    results_file: str | Path | None = None,
    wrapper_kwargs: dict[str, Any] | None = None,
    calibrate_threshold: bool = True,
    prediction_mode: str = "threshold",
    n_thresholds: int = 50,
    save_scores: bool = True,
) -> EvaluationResult:
    """Run one experiment and save result through existing Evaluator."""
    train_data, val_data, test_data = load_experiment_data(
        source=source,
        mode=mode,
        n_shots=n_shots,
        seed=seed,
    )

    kwargs = {"prediction_mode": prediction_mode, **(wrapper_kwargs or {})}
    wrapper = build_wrapper(framework_name, **kwargs)
    quiet_fit = os.environ.get("OOS_QUIET_FIT", "").lower() in ("1", "true", "yes")

    # Measure training time
    t0_train = time.perf_counter()
    if quiet_fit:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            wrapper.fit(train_data["texts"], train_data["labels"])
    else:
        wrapper.fit(train_data["texts"], train_data["labels"])
    train_sec = time.perf_counter() - t0_train

    if prediction_mode == "argmax":
        calibrate_threshold = False

    # Measure calibration time (separate from training)
    calibrate_sec = 0.0
    if calibrate_threshold:
        t0_calib = time.perf_counter()
        threshold = wrapper.calibrate_threshold(
            val_data["texts"],
            val_data["labels"],
            n_thresholds=n_thresholds,
        )
        calibrate_sec = time.perf_counter() - t0_calib
        LOGGER.info(
            "Calibrated threshold for %s (%s/%s): %.4f",
            framework_name,
            source,
            mode if mode == "full" else f"{n_shots}shot_seed{seed}",
            threshold,
        )

    mode_str = "full" if mode == "full" else f"{n_shots}shot"
    result_path = _resolve_results_file(results_file)
    evaluator = Evaluator(test_data=test_data, results_dir=result_path.parent)
    evaluator.metrics_file = result_path
    if quiet_fit:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = evaluator.evaluate(
                model=wrapper,
                model_name=wrapper.model_name,
                mode=mode_str,
                n_shots=n_shots,
                seed=seed,
                source=source,
                save_scores_flag=save_scores,
            )
    else:
        result = evaluator.evaluate(
            model=wrapper,
            model_name=wrapper.model_name,
            mode=mode_str,
            n_shots=n_shots,
            seed=seed,
            source=source,
            save_scores_flag=save_scores,
        )
    result.extra = {
        "framework": framework_name,
        "source": source,
        "prediction_mode": prediction_mode,
        "train_sec": round(train_sec, 2),
        "calibrate_sec": round(calibrate_sec, 2),
        **result.extra,
    }
    if is_degenerate_result(result):
        if framework_name == "h2o" and hasattr(wrapper, "release"):
            wrapper.release()
        raise RuntimeError(
            f"Degenerate predictions for {framework_name} ({mode_str}, seed={seed}): "
            f"in_domain_acc={result.in_domain_acc:.4f}, oos_recall={result.oos_recall:.4f}"
        )
    evaluator.save(result)
    _log_evaluation_result(result)
    if framework_name == "h2o" and hasattr(wrapper, "release"):
        wrapper.release()
    return result


def is_degenerate_result(result: EvaluationResult | dict) -> bool:
    """Detect collapsed predict-all-OOS style metrics."""
    if isinstance(result, EvaluationResult):
        rec = result.to_dict()
    else:
        rec = result
    ind = rec.get("in_domain_acc")
    oos_r = rec.get("oos_recall")
    auroc = rec.get("auroc")
    if ind is not None and oos_r is not None and ind < 0.5 and oos_r > 0.9:
        return True
    if auroc is not None and auroc <= 0.55 and oos_r is not None and oos_r > 0.9:
        return True
    return False


def _log_evaluation_result(result: EvaluationResult) -> None:
    """Print metrics to stdout (compact on Kaggle via OOS_METRICS_LOG=compact)."""
    payload = result.to_dict()
    line = json.dumps(payload, ensure_ascii=False, default=str)
    LOGGER.info("METRICS_RECORD %s", line)
    if os.environ.get("OOS_METRICS_LOG", "").lower() == "compact":
        print("RUN_FINISH " + line)
        return
    print(f"\n{'=' * 72}\nMETRICS_RECORD\n{'=' * 72}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print("=" * 72)


def run_framework_grid(
    frameworks: list[str],
    sources: list[str],
    run_full: bool,
    run_fewshot: bool,
    n_shots_list: list[int],
    seeds: list[int],
    results_file: str | Path | None = None,
    continue_on_error: bool = False,
    wrapper_kwargs: dict[str, Any] | None = None,
    calibrate_threshold: bool = True,
    prediction_mode: str = "threshold",
    save_scores: bool = True,
) -> tuple[list[EvaluationResult], list[dict[str, Any]]]:
    """
    Run framework/source/mode grid. On failure logs metadata and continues.
    """
    if not run_full and not run_fewshot:
        raise ValueError("At least one mode must be enabled: run_full or run_fewshot")

    results: list[EvaluationResult] = []
    errors: list[dict[str, Any]] = []

    jobs: list[dict[str, Any]] = []
    for framework in frameworks:
        for source in sources:
            if run_full:
                jobs.append(
                    {
                        "framework_name": framework,
                        "source": source,
                        "mode": "full",
                        "n_shots": None,
                        "seed": None,
                    }
                )
            if run_fewshot:
                for n_shots in n_shots_list:
                    for seed in seeds:
                        jobs.append(
                            {
                                "framework_name": framework,
                                "source": source,
                                "mode": "fewshot",
                                "n_shots": n_shots,
                                "seed": seed,
                            }
                        )

    for job in jobs:
        try:
            LOGGER.info(
                "Running: framework=%s source=%s mode=%s n_shots=%s seed=%s",
                job["framework_name"],
                job["source"],
                job["mode"],
                job["n_shots"],
                job["seed"],
            )
            result = run_single_experiment(
                results_file=results_file,
                wrapper_kwargs=wrapper_kwargs,
                calibrate_threshold=calibrate_threshold,
                prediction_mode=prediction_mode,
                save_scores=save_scores,
                **job,
            )
            results.append(result)
        except Exception as exc:
            error_payload = {
                "framework": job["framework_name"],
                "source": job["source"],
                "mode": job["mode"],
                "n_shots": job["n_shots"],
                "seed": job["seed"],
                "error": str(exc),
            }
            errors.append(error_payload)
            LOGGER.exception("Experiment failed: %s", error_payload)
            if not continue_on_error:
                raise

    return results, errors
