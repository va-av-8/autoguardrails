from __future__ import annotations

import logging
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
    n_thresholds: int = 50,
) -> EvaluationResult:
    """Run one experiment and save result through existing Evaluator."""
    train_data, val_data, test_data = load_experiment_data(
        source=source,
        mode=mode,
        n_shots=n_shots,
        seed=seed,
    )

    wrapper = build_wrapper(framework_name, **(wrapper_kwargs or {}))
    wrapper.fit(train_data["texts"], train_data["labels"])

    if calibrate_threshold:
        threshold = wrapper.calibrate_threshold(
            val_data["texts"],
            val_data["labels"],
            n_thresholds=n_thresholds,
        )
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
    result = evaluator.evaluate(
        model=wrapper,
        model_name=wrapper.model_name,
        mode=mode_str,
        n_shots=n_shots,
        seed=seed,
    )
    result.extra = {"framework": framework_name, "source": source, **result.extra}
    evaluator.save(result)
    return result


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
