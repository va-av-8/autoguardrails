"""
Запуск AutoIntent на WildJailbreak (обучение + оценка).

Использование:
    # Pilot (маленький embedder, быстрая валидация)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --pilot

    # Final (оригинальный embedder e5, preset classic-light по умолчанию)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42

    # Другой preset (classic-medium, nn-medium, zero-shot-encoders и др.)
    python scripts/run_autointent.py --preset classic-medium --mode fewshot --n_shots 10 --seed 42

    # AutoML без фиксации embedder (медленнее, потенциально лучше)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --no-fix-embedder

    # Только оценка (модель должна существовать)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --eval-only

    # Full train (100K stratified wildjailbreak_full100k_seed{S}.json из prepare_data --full_subset)
    python scripts/run_autointent.py --mode full --seed 42

    # Другой search-space пресет (classic-medium, nn-medium, zero-shot-encoders, transformers-light, …)
    python scripts/run_autointent.py --mode fewshot --n_shots 20 --seed 42 --preset classic-medium

Данные:
    - Few-shot train: data/processed/train_shot{N}_seed{S}.json
    - Full train: data/processed/wildjailbreak_full100k_seed{S}.json
      (это 100K стратифицированная подвыборка, не весь сырой WildJailbreak train;
      готовится скриптом prepare_data.py --full_subset из‑за размера и времени AutoML)
    - Test: data/processed/test.json (для inference)
    - Eval binary: data/processed/wildjailbreak_eval_binary.jsonl (для метрик)

Результаты:
    - Каталоги runs: имя включает пресет, напр.
      `runs/autointent_classic-medium_{N}shot_seed{S}/`,
      `runs/autointent_transformers-light_autoembedder_full_seed{S}/`
    - Метрики: results/metrics.json (числовые поля + extra: embedder, eval_counts TP/FP/FN/TN,
      scoring_module_attrs, decision_module_attrs, scores_eval_summary, model_dir и т.д.)
    - На каждый прогон оценки: runs/metrics_<model_name>_<mode>_seed<S>.json — тот же JSON-объект, что одна строка в metrics.json
    - Полные скоры eval: runs/eval_scores_<model_name>_<mode>_seed<S>.jsonl — для анализа ошибок
    - Дублирование метрик в stdout: --print-metrics-json (Kaggle/логи, если файлы не сохранились)
    - На Kaggle после каждого append в metrics.json делается копия в
      /kaggle/working/metrics_jailbreak_latest.json (удобно до zip / конца сессии).

    Пример: few-shot без фиксации эмбеддера на всех сидах:
        python scripts/run_autointent.py --mode fewshot --n_shots 10 --no-fix-embedder --all-seeds

Стабильность на macOS ARM:
    load_dotenv и лимиты потоков BLAS/OpenMP должны применяться до import numpy/torch.
    Скопируйте .env.example в .env при segfault/sklearn warnings во время AutoML.

Прогресс AutoML (по умолчанию):
    полоса Optuna (tqdm) по числу trials в текущем узле (scoring/decision);
    строка «Overall HPO (estimate)» — грубая доля по этапам HPO (не покрывает долгий первый embed внутри trial).
    Отключить: --no-automl-progress

Kaggle / меньше «красного» stderr:
    при наличии /kaggle/working (или JAILBREAK_QUIET_LOGS=1) поднимается уровень root-log,
    глушатся sentence_transformers/transformers/optuna и tqdm через env (см. _apply_kaggle_quiet_env).
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import logging
import os
import platform
import shutil
import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Paths before dotenv / numpy (thread env must be set before BLAS loads).
script_dir = Path(__file__).resolve().parent
task_dir = script_dir.parent
project_root = task_dir.parent.parent

from dotenv import load_dotenv

load_dotenv(project_root / ".env")


def _apply_thread_env_defaults() -> None:
    """Single-thread BLAS/OpenMP to reduce crashes on macOS (especially Apple Silicon)."""
    defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "LOKY_MAX_CPU_COUNT": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in defaults.items():
        if key not in os.environ or os.environ[key] == "":
            os.environ[key] = value


def _apply_kaggle_quiet_env() -> None:
    """Меньше HF/tqdm в stderr на Kaggle; TRANSFORMERS_* до import torch/transformers."""
    if not Path("/kaggle/working").exists() and os.environ.get("JAILBREAK_QUIET_LOGS") != "1":
        return
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")


_apply_thread_env_defaults()
_apply_kaggle_quiet_env()

import numpy as np
import torch

sys.path.insert(0, str(project_root))

from autointent import Pipeline, Dataset as AIDataset
from autointent.configs import LoggingConfig, EmbedderConfig, DataConfig

from tasks.jailbreak_detection.src.metrics import evaluate_jailbreak

logger = logging.getLogger(__name__)

# Имена search-space пресетов AutoIntent (autointent==0.2.0, см. SearchSpacePreset).
SEARCH_SPACE_PRESETS: tuple[str, ...] = (
    "classic-heavy",
    "classic-light",
    "classic-medium",
    "nn-heavy",
    "nn-medium",
    "transformers-heavy",
    "transformers-light",
    "transformers-no-hpo",
    "zero-shot-llm",
    "zero-shot-encoders",
)

# CLI-алиасы (имя в отчётах / model_name) → фактический SearchSpacePreset AutoIntent.
PRESET_ALIASES: dict[str, str] = {
    "bert-finetune": "transformers-light",
}

# Все допустимые значения --preset (включая алиасы).
CLI_PRESET_CHOICES: tuple[str, ...] = tuple(
    sorted({*SEARCH_SPACE_PRESETS, *PRESET_ALIASES.keys()})
)

# Пресеты, где scoring использует intent.description (zero-shot encoders / LLM).
PRESETS_REQUIRING_INTENT_DESCRIPTIONS: frozenset[str] = frozenset(
    {"zero-shot-encoders", "zero-shot-llm"}
)

# Описания классов для AutoIntent Intent.description (binary safe/jailbreak).
JAILBREAK_BINARY_INTENTS: tuple[dict[str, Any], ...] = (
    {
        "id": 0,
        "name": "safe",
        "description": (
            "A benign user request that should not be classified as a jailbreak "
            "or harmful instruction."
        ),
    },
    {
        "id": 1,
        "name": "jailbreak",
        "description": (
            "A harmful, adversarial, or jailbreak-style user request attempting to "
            "bypass safety rules or elicit unsafe behavior."
        ),
    },
)


def resolve_preset(preset: str) -> tuple[str, str]:
    """(имя для логов/metrics, имя для Pipeline.from_preset)."""
    cli_preset = preset.strip()
    autointent_preset = PRESET_ALIASES.get(cli_preset, cli_preset)
    if autointent_preset not in SEARCH_SPACE_PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. "
            f"Choose from: {', '.join(CLI_PRESET_CHOICES)}"
        )
    return cli_preset, autointent_preset


def preset_needs_intent_descriptions(cli_preset: str) -> bool:
    _, autointent_preset = resolve_preset(cli_preset)
    return autointent_preset in PRESETS_REQUIRING_INTENT_DESCRIPTIONS


def build_binary_intents(*, with_descriptions: bool) -> list[dict[str, Any]]:
    """AutoIntent intents для binary safe=0 / jailbreak=1."""
    intents: list[dict[str, Any]] = []
    for spec in JAILBREAK_BINARY_INTENTS:
        row: dict[str, Any] = {"id": spec["id"], "name": spec["name"]}
        if with_descriptions:
            row["description"] = spec["description"]
        intents.append(row)
    return intents


def metrics_row_for_export(result: dict) -> dict[str, Any]:
    """Компактная строка метрик для сводных JSON (few-shot grid)."""
    ex = result.get("extra") or {}
    return {
        "model_name": result.get("model_name"),
        "preset": ex.get("preset") or ex.get("search_space_preset"),
        "search_space_preset": ex.get("search_space_preset"),
        "mode": result.get("mode"),
        "n_shots": result.get("n_shots"),
        "seed": result.get("seed"),
        "f1": result.get("f1"),
        "precision": result.get("precision"),
        "recall": result.get("recall"),
        "over_refusal_rate": result.get("over_refusal_rate"),
        "recall_vanilla_harmful": result.get("recall_vanilla_harmful"),
        "recall_adversarial_harmful": result.get("recall_adversarial_harmful"),
        "embedder_hf_model": ex.get("embedder_hf_model"),
        "model_dir": ex.get("model_dir"),
    }


def _configure_run_logging() -> None:
    """Один раз настраивает stderr-logging для сообщений об эмбеддере и др."""
    root = logging.getLogger()
    if root.handlers:
        return
    quiet = Path("/kaggle/working").exists() or os.environ.get("JAILBREAK_QUIET_LOGS") == "1"
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _quiet_third_party_loggers() -> None:
    """Глушит чужие INFO в stderr (Kaggle подсвечивает stderr красным)."""
    if not Path("/kaggle/working").exists() and os.environ.get("JAILBREAK_QUIET_LOGS") != "1":
        return
    for name in (
        "sentence_transformers",
        "transformers",
        "transformers.models",
        "huggingface_hub",
        "datasets",
        "optuna",
        "httpx",
        "httpcore",
        "urllib3",
        "filelock",
        "torch",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger("autointent").setLevel(logging.WARNING)


def _mirror_metrics_to_kaggle_working(metrics_path: Path) -> None:
    """Копия metrics.json в корень /kaggle/working после каждого прогона (видна в Output)."""
    kw = Path("/kaggle/working")
    if not kw.exists() or not metrics_path.is_file():
        return
    dst = kw / "metrics_jailbreak_latest.json"
    shutil.copy2(metrics_path, dst)
    print(f"[metrics] Kaggle snapshot → {dst}", flush=True)


# Совпадает с prepare_data.DEFAULT_SEEDS и именами wildjailbreak_full100k_seed{S}.json
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456)


@dataclass
class _HpoProgressState:
    """Tracks Optuna node stages for coarse overall % (classic-light: scoring + decision)."""

    node_total: int = 1
    node_index: int = 0


def _make_overall_hpo_callback(n_trials: int | None, state: _HpoProgressState):
    """Optuna callback: print estimated overall HPO % across NodeOptimizer stages."""

    def overall_callback(study, trial) -> None:
        t = trial.number + 1
        denom = n_trials if n_trials is not None else max(t, 1)
        span = 100.0 / max(state.node_total, 1)
        base = (state.node_index / max(state.node_total, 1)) * 100.0
        frac = min(t / float(max(denom, 1)), 1.0)
        overall = base + frac * span
        print(
            f"\r[AutoIntent] Overall HPO (estimate): {overall:5.1f}%  —  "
            f"trial {t}/{denom} in current stage    ",
            end="",
            flush=True,
        )

    return overall_callback


def _install_automl_progress_hooks(
    pipeline: Pipeline,
    *,
    enable_optuna_bar: bool,
) -> Callable[[], None]:
    """
    Optuna hides progress by default; AutoIntent sets WARNING verbosity.
    We enable tqdm per study.optimize and print coarse overall % across HPO stages.
    """
    from autointent.nodes._node_optimizer import NodeOptimizer

    import autointent.nodes._node_optimizer as _no_mod
    import optuna

    state = _HpoProgressState()
    state.node_total = max(
        1,
        sum(1 for n in pipeline.nodes.values() if isinstance(n, NodeOptimizer)),
    )

    Study = optuna.study.Study
    _orig_optimize = Study.optimize

    def _wrapped_optimize(self, func, *args, **kwargs):
        if enable_optuna_bar:
            kwargs.setdefault("show_progress_bar", True)
        n_trials_kw = kwargs.get("n_trials")
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(_make_overall_hpo_callback(n_trials_kw, state))
        kwargs["callbacks"] = callbacks
        return _orig_optimize(self, func, *args, **kwargs)

    _orig_node_fit = NodeOptimizer.fit

    def _wrapped_node_fit(self, context):
        print()
        stage = state.node_index + 1
        label = self.node_info.node_type.value
        nt = context.hpo_config.n_trials
        print(
            f"[AutoIntent] HPO stage {stage}/{state.node_total}: {label!r} "
            f"(up to {nt} Optuna trials per stage; bar = trials in this stage)",
            flush=True,
        )
        try:
            return _orig_node_fit(self, context)
        finally:
            state.node_index += 1

    Study.optimize = _wrapped_optimize  # type: ignore[method-assign]
    _no_mod.NodeOptimizer.fit = _wrapped_node_fit  # type: ignore[method-assign]

    def _restore() -> None:
        Study.optimize = _orig_optimize  # type: ignore[method-assign]
        _no_mod.NodeOptimizer.fit = _orig_node_fit  # type: ignore[method-assign]

    return _restore


def full_train_filename(seed: int) -> str:
    return f"wildjailbreak_full100k_seed{seed}.json"


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


def embedder_hf_model_from_dump(model_dir: Path) -> str | None:
    """HF id эмбеддера после pipeline.dump (актуально для --no-fix-embedder)."""
    p = model_dir / "scoring_module" / "pydantic" / "embedder_config" / "model_dump.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    name = data.get("model_name")
    return name if isinstance(name, str) and name.strip() else None


def decision_module_attrs_from_dump(model_dir: Path) -> dict[str, Any] | None:
    """Атрибуты decision-модуля из dump (порог и др.) для логов metrics.json."""
    p = model_dir / "decision_module" / "simple_attrs.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _json_size_limited(value: Any, *, max_chars: int = 12_000) -> Any:
    """Return value if JSON-serializable and small enough, otherwise a compact descriptor."""
    try:
        dumped = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return {"_error": "non_serializable"}
    if len(dumped) <= max_chars:
        return value
    if isinstance(value, dict):
        return {
            "_truncated": True,
            "json_chars": len(dumped),
            "top_level_keys": list(value.keys()),
        }
    if isinstance(value, list):
        return {
            "_truncated": True,
            "json_chars": len(dumped),
            "type": "list",
            "length": len(value),
        }
    return {
        "_truncated": True,
        "json_chars": len(dumped),
        "type": type(value).__name__,
    }


def _file_info(path: Path) -> dict[str, Any]:
    """Small reproducibility descriptor for data/model files."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
    }


def _basic_text_stats(texts: list[str]) -> dict[str, Any]:
    lengths = np.asarray([len(t) for t in texts], dtype=np.float64)
    if len(lengths) == 0:
        return {"n": 0}
    return {
        "n": int(len(texts)),
        "chars_mean": float(np.mean(lengths)),
        "chars_std": float(np.std(lengths)),
        "chars_min": int(np.min(lengths)),
        "chars_p50": float(np.percentile(lengths, 50)),
        "chars_p95": float(np.percentile(lengths, 95)),
        "chars_max": int(np.max(lengths)),
    }


def summarize_train_split(train_raw: dict, *, source_file: Path) -> dict[str, Any]:
    """Summary of AutoIntent train split before conversion."""
    safe = []
    for intent in train_raw.get("intents", []):
        safe.extend(intent.get("utterances", []))
    jailbreak = list(train_raw.get("oos_utterances", []))
    total = len(safe) + len(jailbreak)
    return {
        "source_file": _file_info(source_file),
        "n_total": total,
        "n_safe": len(safe),
        "n_jailbreak": len(jailbreak),
        "class_balance_safe": float(len(safe) / total) if total else None,
        "class_balance_jailbreak": float(len(jailbreak) / total) if total else None,
        "safe_text_stats": _basic_text_stats(safe),
        "jailbreak_text_stats": _basic_text_stats(jailbreak),
    }


def summarize_test_split(test_raw: dict, *, source_file: Path) -> dict[str, Any]:
    """Summary of AutoIntent test split used for inference."""
    texts = list(test_raw.get("utterances", []))
    labels = np.asarray(test_raw.get("labels", []))
    return {
        "source_file": _file_info(source_file),
        "n_total": int(len(texts)),
        "n_safe": int(np.sum(labels == 0)),
        "n_jailbreak": int(np.sum(labels == 1)),
        "text_stats": _basic_text_stats(texts),
    }


def summarize_eval_binary(eval_binary: list[dict], *, source_file: Path) -> dict[str, Any]:
    """Summary of evaluation labels and WildJailbreak data_type breakdown."""
    by_label: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in eval_binary:
        label = str(row.get("binary_label"))
        dtype = str(row.get("data_type"))
        by_label[label] = by_label.get(label, 0) + 1
        by_type[dtype] = by_type.get(dtype, 0) + 1
    return {
        "source_file": _file_info(source_file),
        "n_total": len(eval_binary),
        "by_binary_label": dict(sorted(by_label.items())),
        "by_data_type": dict(sorted(by_type.items())),
    }


def prediction_distribution_by_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_types: np.ndarray,
) -> dict[str, Any]:
    """Per-data_type counts and rates, useful for jailbreak vs benign failure analysis."""
    out: dict[str, Any] = {}
    for dtype in sorted({str(x) for x in data_types.tolist()}):
        mask = data_types == dtype
        yt = y_true[mask]
        yp = y_pred[mask]
        n = int(np.sum(mask))
        pred_jb = int(np.sum(yp == 1))
        true_jb = int(np.sum(yt == 1))
        out[dtype] = {
            "n": n,
            "true_safe": int(np.sum(yt == 0)),
            "true_jailbreak": true_jb,
            "pred_safe": int(np.sum(yp == 0)),
            "pred_jailbreak": pred_jb,
            "pred_jailbreak_rate": float(pred_jb / n) if n else None,
            "correct": int(np.sum(yt == yp)),
            "accuracy": float(np.mean(yt == yp)) if n else None,
        }
    return out


def runtime_environment_summary() -> dict[str, Any]:
    """Runtime details for Kaggle/local reproducibility without logging secrets."""
    packages = {}
    for pkg in (
        "autointent",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "sentence-transformers",
        "datasets",
        "numpy",
        "scikit-learn",
        "optuna",
    ):
        try:
            packages[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            packages[pkg] = None
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
        "env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "LOKY_MAX_CPU_COUNT",
                "TOKENIZERS_PARALLELISM",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "kaggle": {
            "is_kaggle": Path("/kaggle/working").exists(),
            "working_exists": Path("/kaggle/working").exists(),
            "input_exists": Path("/kaggle/input").exists(),
        },
        "packages": packages,
    }


def pipeline_artifact_manifest(model_dir: Path) -> dict[str, Any]:
    """Compact listing of saved AutoIntent modules/configs."""
    manifest: dict[str, Any] = {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "top_level_entries": [],
        "json_files": [],
    }
    if not model_dir.exists():
        return manifest
    top_entries = []
    for p in sorted(model_dir.iterdir(), key=lambda x: x.name):
        item = {"name": p.name, "type": "dir" if p.is_dir() else "file"}
        if p.is_file():
            item.update({"bytes": p.stat().st_size})
        top_entries.append(item)
    manifest["top_level_entries"] = top_entries

    json_files = []
    for p in sorted(model_dir.rglob("*.json"))[:80]:
        rel = p.relative_to(model_dir)
        info = _file_info(p)
        info["relative_path"] = str(rel)
        json_files.append(info)
    manifest["json_files"] = json_files
    return manifest
def scoring_module_attrs_from_dump(model_dir: Path) -> dict[str, Any] | None:
    """Атрибуты scoring-модуля из dump (k, weights и др.) для логов metrics.json."""
    p = model_dir / "scoring_module" / "simple_attrs.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def confusion_and_rates_jailbreak_positive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_label: int = 1,
) -> dict[str, Any]:
    """TP/FP/FN/TN и явные FNR/FPR для jailbreak=positive (для cost / HYP-JB-004)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    pos = positive_label
    tp = int(np.sum((yt == pos) & (yp == pos)))
    fp = int(np.sum((yt != pos) & (yp == pos)))
    fn = int(np.sum((yt == pos) & (yp != pos)))
    tn = int(np.sum((yt != pos) & (yp != pos)))
    denom_p = tp + fn
    denom_n = fp + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fnr_jailbreak": float(fn / denom_p) if denom_p > 0 else None,
        "fpr_safe": float(fp / denom_n) if denom_n > 0 else None,
        "n_eval": int(len(yt)),
        "n_safe_true": int(np.sum(yt != pos)),
        "n_jailbreak_true": int(np.sum(yt == pos)),
    }


def scoring_eval_summary_from_pipeline(
    pipeline: Pipeline,
    texts: list[str],
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    """
    Краткая статистика по скорам scoring (margin между классами 0 и 1).
    Предполагается порядок классов как в train: 0=safe, 1=jailbreak.

    Returns:
        (summary_dict, scores_array) — summary для metrics.json и полный массив скоров.
    """
    try:
        inf_out = pipeline.predict_with_metadata(texts)
        utterances = getattr(inf_out, "utterances", None)
        if not isinstance(utterances, list) or not utterances:
            return None, None
        rows = []
        for u in utterances:
            sc = getattr(u, "score", None)
            if sc is None:
                return None, None
            rows.append(np.asarray(sc, dtype=np.float64).ravel())
        scores = np.stack(rows, axis=0)
    except Exception as exc:
        logger.debug("scoring_eval_summary_from_pipeline: %s", exc)
        return None, None
    if scores.ndim != 2 or scores.shape[1] < 2:
        return {
            "note": "unexpected_score_shape",
            "shape": list(scores.shape),
        }, None
    margin = scores[:, 1] - scores[:, 0]
    summary = {
        "n_scored": int(scores.shape[0]),
        "n_score_dims": int(scores.shape[1]),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
        "margin_min": float(np.min(margin)),
        "margin_max": float(np.max(margin)),
        "score_col0_mean": float(np.mean(scores[:, 0])),
        "score_col1_mean": float(np.mean(scores[:, 1])),
        "note": "col0≈safe intent col1≈jailbreak (binary scoring)",
    }
    return summary, scores


def save_eval_scores(
    output_path: Path,
    metadata: dict[str, Any],
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> None:
    """
    Сохраняет полные скоры eval прогона в JSONL для анализа ошибок.

    Формат каждой строки:
    {"model_name", "preset", "mode", "n_shots", "seed", "idx", "text", "y_true", "y_pred", "score_safe", "score_jb"}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (text, yt, yp) in enumerate(zip(texts, y_true, y_pred)):
            row = {
                "model_name": metadata.get("model_name", ""),
                "preset": metadata.get("preset", ""),
                "mode": metadata.get("mode", ""),
                "n_shots": metadata.get("n_shots"),
                "seed": metadata.get("seed"),
                "idx": idx,
                "text": text,
                "y_true": int(yt),
                "y_pred": int(yp),
                "score_safe": float(scores[idx, 0]),
                "score_jb": float(scores[idx, 1]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved eval scores to %s (%d rows)", output_path, len(texts))


def get_model_name(preset: str, pilot: bool, no_fix_embedder: bool) -> str:
    """Формирует имя модели на основе пресета и флагов."""
    base = f"autointent_{preset}"
    if no_fix_embedder:
        return f"{base}_autoembedder"
    if pilot:
        return f"{base}_pilot"
    return base


def get_embedder_name(pilot: bool) -> str:
    if pilot:
        return "intfloat/multilingual-e5-small"
    return "intfloat/multilingual-e5-large-instruct"


def make_query_prompt_suffix(query_prompt: str | None, max_len: int = 32) -> str:
    """
    Build a filename suffix from query_prompt.

    Returns "" if query_prompt is None/empty (backward compatibility).
    Otherwise returns "_qp_<slug>" where slug is lowercase alphanumeric.
    """
    if not query_prompt:
        return ""
    slug = query_prompt.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return f"_qp_{slug}" if slug else ""


def get_model_dir(
    preset: str,
    pilot: bool,
    no_fix_embedder: bool,
    mode: str,
    n_shots: int | None,
    seed: int,
    query_prompt: str | None = None,
) -> Path:
    model_name = get_model_name(preset, pilot, no_fix_embedder)
    qp_suffix = make_query_prompt_suffix(query_prompt)
    if mode == "full":
        return get_runs_dir() / f"{model_name}_full_seed{seed}{qp_suffix}"
    assert n_shots is not None
    return get_runs_dir() / f"{model_name}_{n_shots}shot_seed{seed}{qp_suffix}"


def load_fewshot_train(n_shots: int, seed: int, data_dir: Path) -> dict:
    """
    Load few-shot train data in AutoIntent format.
    Format: {"intents": [{"id": 0, "name": "safe", "utterances": [...]}],
             "oos_utterances": [...]}
    """
    path = data_dir / f"train_shot{n_shots}_seed{seed}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_full_train(seed: int, data_dir: Path) -> dict:
    """Full-train subset in same AutoIntent dict format as few-shot (prepare_data --full_subset)."""
    path = data_dir / full_train_filename(seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"Full-train file missing: {path}\n"
            "Create 100K stratified splits with:\n"
            "  python tasks/jailbreak_detection/scripts/prepare_data.py --full_subset\n"
            "(HF gated dataset + token, or place raw jsonl under data/raw/; see prepare_data.py.)"
        )
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

    # Remove existing entry with same model_name, mode, n_shots, seed, query_prompt
    all_results = [
        r for r in all_results
        if not (
            r.get("model_name") == result["model_name"] and
            r.get("mode") == result["mode"] and
            r.get("n_shots") == result.get("n_shots") and
            r.get("seed") == result.get("seed") and
            r.get("query_prompt") == result.get("query_prompt")
        )
    ]

    all_results.append(result)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved to {metrics_path}")
    _mirror_metrics_to_kaggle_working(metrics_path)


def run_metrics_filename(result: dict) -> str:
    """Stable filename for per-run metrics JSON under runs/ (one object = one metrics.json row)."""
    mn = result["model_name"].replace("/", "_").replace(" ", "_")
    mode = result["mode"]
    seed = result["seed"]
    qp_suffix = make_query_prompt_suffix(result.get("query_prompt"))
    if mode == "full":
        return f"metrics_{mn}_full_seed{seed}{qp_suffix}.json"
    return f"metrics_{mn}_{mode}_seed{seed}{qp_suffix}.json"


def save_run_metrics_file(result: dict, runs_dir: Path) -> Path:
    """Write a single run result under runs/; schema matches one element appended to metrics.json."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / run_metrics_filename(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Run metrics saved to {path}")
    return path


def train(args, data_dir: Path, model_dir: Path) -> None:
    """Train AutoIntent pipeline."""
    train_started_at = time.perf_counter()
    cli_preset, autointent_preset = resolve_preset(getattr(args, "preset", "classic-light"))
    args.preset = cli_preset
    no_fix_embedder = getattr(args, "no_fix_embedder", False)
    mode = getattr(args, "mode", "fewshot")
    if no_fix_embedder:
        embedder_name = "auto (optimized by AutoML)"
    else:
        embedder_name = get_embedder_name(args.pilot)

    print("=" * 60)
    print("AutoIntent Training (Jailbreak Detection)")
    print("=" * 60)
    if no_fix_embedder:
        mode_label = "AUTO-EMBEDDER"
    else:
        mode_label = "PILOT" if args.pilot else "FINAL"
    print(f"Mode: {mode_label}")
    print(f"Search-space preset (CLI): {cli_preset}")
    if autointent_preset != cli_preset:
        print(f"AutoIntent preset: {autointent_preset}")
    print(f"Embedder: {embedder_name}")
    if no_fix_embedder:
        logger.info(
            "Embedder policy: AutoML (search-space preset %s / %s); "
            "HF model_name после dump.",
            cli_preset,
            autointent_preset,
        )
    else:
        logger.info(
            "Embedder policy: фиксированный; HuggingFace model_name=%s",
            embedder_name,
        )
    if mode == "full":
        print(f"Training: FULL ({full_train_filename(args.seed)}), seed={args.seed}")
    else:
        print(f"Training: {args.n_shots}shot, seed={args.seed}")
    print(f"Output: {model_dir}")
    print("=" * 60)
    print()

    # Load data
    if mode == "full":
        train_source_file = data_dir / full_train_filename(args.seed)
        train_raw = load_full_train(args.seed, data_dir)
    else:
        train_source_file = data_dir / f"train_shot{args.n_shots}_seed{args.seed}.json"
        train_raw = load_fewshot_train(args.n_shots, args.seed, data_dir)
    test_source_file = data_dir / "test.json"
    test_raw = load_test(data_dir)

    train_ai = convert_to_autointent_train(train_raw)
    test_ai = convert_to_autointent_test(test_raw)
    train_data_summary = summarize_train_split(train_raw, source_file=train_source_file)
    test_data_summary = summarize_test_split(test_raw, source_file=test_source_file)

    # Binary classification: 2 intents (no OOS); descriptions обязательны для zero-shot-encoders
    with_descriptions = preset_needs_intent_descriptions(cli_preset)
    intents = build_binary_intents(with_descriptions=with_descriptions)
    if with_descriptions:
        print("Intent descriptions: enabled (required for zero-shot encoder presets)")

    n_safe = len(train_raw["intents"][0]["utterances"])
    n_jailbreak = len(train_raw["oos_utterances"])
    print(f"Train: {len(train_ai)} samples")
    print(f"  - safe (label=0): {n_safe}")
    print(f"  - jailbreak (label=1): {n_jailbreak}")
    print(f"Test: {len(test_ai)} samples")
    print(f"Train source: {train_source_file}")
    print(f"Test source: {test_source_file}")
    print()

    # Create AutoIntent Dataset
    ai_dataset = AIDataset.from_dict({
        "train": train_ai,
        "test": test_ai,
        "intents": intents,
    })

    # Create pipeline with selected preset
    preset = args.preset
    print(f"Using preset: {preset}")
    pipeline = Pipeline.from_preset(autointent_preset, seed=args.seed)
    if not no_fix_embedder:
        embedder_kwargs: dict[str, Any] = {"model_name": embedder_name}
        if getattr(args, "query_prompt", None) is not None:
            embedder_kwargs["query_prompt"] = args.query_prompt
        pipeline.set_config(EmbedderConfig(**embedder_kwargs))

    # Cross-validation only for few-shot (full train uses default scheme)
    if mode == "fewshot":
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

    show_progress = not getattr(args, "no_automl_progress", False)
    restore_hooks: Callable[[], None] | None = None
    try:
        if show_progress:
            restore_hooks = _install_automl_progress_hooks(
                pipeline,
                enable_optuna_bar=True,
            )
        pipeline.fit(ai_dataset)
    finally:
        if restore_hooks is not None:
            print()
            restore_hooks()

    print()
    print("AutoML optimization completed!")

    # Save model
    print(f"Saving model to {model_dir}...")
    pipeline.dump(model_dir)

    # Save metadata
    if mode == "full":
        meta_mode = "full"
        meta_n_shots = None
    else:
        meta_mode = f"{args.n_shots}shot"
        meta_n_shots = args.n_shots

    metadata = {
        "model_name": get_model_name(args.preset, args.pilot, no_fix_embedder),
        "mode": meta_mode,
        "n_shots": meta_n_shots,
        "seed": args.seed,
        "query_prompt": getattr(args, "query_prompt", None),
        "embedder": embedder_name,
        "embedder_fixed": not no_fix_embedder,
        "pilot": args.pilot,
        "preset": args.preset,
        "task": "jailbreak_detection",
        "approach": "binary_classification",  # Not OOS detection
    }
    resolved_hf = embedder_hf_model_from_dump(model_dir)
    if resolved_hf:
        metadata["embedder_hf_model"] = resolved_hf
        print(f"Chosen HuggingFace embedder (from dump): {resolved_hf}")
        logger.info(
            "В сохранённом пайплайне эмбеддер (HF model_name): %s",
            resolved_hf,
        )
    else:
        logger.warning(
            "Не удалось прочитать HF model_name из "
            "scoring_module/pydantic/embedder_config/model_dump.json под %s",
            model_dir,
        )

    train_elapsed_sec = time.perf_counter() - train_started_at
    metadata["data_summary"] = {
        "train": train_data_summary,
        "test": test_data_summary,
    }
    metadata["run_config"] = {
        "mode": mode,
        "n_shots": args.n_shots if mode == "fewshot" else None,
        "seed": args.seed,
        "preset": cli_preset,
        "autointent_preset": autointent_preset,
        "pilot": args.pilot,
        "no_fix_embedder": no_fix_embedder,
        "data_config": (
            {"scheme": "cv", "n_folds": 3}
            if mode == "fewshot"
            else {"scheme": "default_autointent"}
        ),
        "logging_config": {
            "project_dir": str(model_dir),
            "dump_modules": True,
            "clear_ram": False,
        },
    }
    metadata["runtime_environment"] = runtime_environment_summary()
    metadata["timings"] = {
        "train_total_sec": float(train_elapsed_sec),
    }

    (model_dir / "train_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Training completed!")


def evaluate(
    args,
    data_dir: Path,
    model_dir: Path,
    results_dir: Path,
    runs_dir: Path | None = None,
) -> None:
    """Evaluate trained AutoIntent pipeline."""
    eval_started_at = time.perf_counter()
    print()
    print("=" * 60)
    print("AutoIntent Evaluation (Jailbreak Detection)")
    print("=" * 60)

    # Load metadata
    metadata_path = model_dir / "train_metadata.json"
    no_fix_embedder = getattr(args, "no_fix_embedder", False)
    cli_preset, autointent_preset = resolve_preset(getattr(args, "preset", "classic-light"))
    mode = getattr(args, "mode", "fewshot")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        # Ensure preset is set (for backward compatibility with old train_metadata.json)
        if "preset" not in metadata:
            metadata["preset"] = args.preset
    else:
        if mode == "full":
            meta_mode = "full"
            meta_n_shots = None
        else:
            meta_mode = f"{args.n_shots}shot"
            meta_n_shots = args.n_shots
        metadata = {
            "model_name": get_model_name(args.preset, args.pilot, no_fix_embedder),
            "mode": meta_mode,
            "n_shots": meta_n_shots,
            "seed": args.seed,
            "embedder": (
                "auto (optimized by AutoML)"
                if no_fix_embedder
                else get_embedder_name(args.pilot)
            ),
            "embedder_fixed": not no_fix_embedder,
            "pilot": args.pilot,
            "preset": args.preset,
        }

    hf_emb = metadata.get("embedder_hf_model") or embedder_hf_model_from_dump(model_dir)

    print(f"Model: {metadata['model_name']}")
    print(f"Mode: {metadata['mode']}")
    print(f"Embedder: {metadata['embedder']}")
    if hf_emb:
        print(f"HuggingFace embedder model: {hf_emb}")
        logger.info("Оценка: используется HF эмбеддер model_name=%s", hf_emb)
    elif metadata.get("embedder_fixed", True):
        logger.info(
            "Оценка: по метаданным фиксированный режим; HF model из metadata/dump не повторён.",
        )
    else:
        logger.warning(
            "Оценка: autoembedder, но HF model_name не найден в metadata и dump (%s)",
            model_dir,
        )
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model from {model_dir}...")
    pipeline = Pipeline.load(model_dir)
    print("Model loaded!")
    print()

    # Load test data
    test_source_file = data_dir / "test.json"
    eval_binary_source_file = data_dir / "wildjailbreak_eval_binary.jsonl"
    test_raw = load_test(data_dir)
    test_texts = test_raw["utterances"]

    # Load eval binary for data_types and ground truth
    eval_binary = load_eval_binary(data_dir)
    y_true = np.array([1 if r["binary_label"] == "jailbreak" else 0 for r in eval_binary])
    data_types = np.array([r["data_type"] for r in eval_binary])
    test_data_summary = summarize_test_split(test_raw, source_file=test_source_file)
    eval_binary_summary = summarize_eval_binary(
        eval_binary,
        source_file=eval_binary_source_file,
    )
    data_alignment = {
        "test_utterances": len(test_texts),
        "test_labels": len(test_raw.get("labels", [])),
        "eval_binary_rows": len(eval_binary),
        "aligned_lengths": len(test_texts) == len(test_raw.get("labels", [])) == len(eval_binary),
    }

    print(f"Test samples: {len(test_texts)}")
    print(f"  safe: {sum(y_true == 0)}, jailbreak: {sum(y_true == 1)}")
    print(f"Data alignment OK: {data_alignment['aligned_lengths']}")
    print()
    if not data_alignment["aligned_lengths"]:
        raise ValueError(f"Test/eval data length mismatch: {data_alignment}")

    # Predictions
    print("Running predictions...")
    raw_preds = pipeline.predict(test_texts)

    # Binary classification: predictions are 0 (safe) or 1 (jailbreak) directly
    # None means model couldn't decide - treat as jailbreak (safer for guardrail)
    raw_pred_none = int(sum(p is None for p in raw_preds))
    raw_pred_values: dict[str, int] = {}
    for pred in raw_preds:
        key = "None" if pred is None else str(pred)
        raw_pred_values[key] = raw_pred_values.get(key, 0) + 1
    y_pred = np.array([1 if p is None else p for p in raw_preds])

    print(f"Predictions: {len(y_pred)}")
    print(f"  predicted safe: {sum(y_pred == 0)}")
    print(f"  predicted jailbreak: {sum(y_pred == 1)}")
    print(f"  undecided/None treated as jailbreak: {raw_pred_none}")
    print()

    # Compute metrics                                                                                                                                                                 
    print("Computing metrics...")                                                                                                                                                     
    metrics = evaluate_jailbreak(y_true, y_pred, data_types, oos_label=1)                                                                                                             

    eval_counts = confusion_and_rates_jailbreak_positive(y_true, y_pred, positive_label=1)                                                                                            
    prediction_by_type = prediction_distribution_by_type(y_true, y_pred, data_types)                                                                                                  
    decision_attrs_json = _json_size_limited(decision_module_attrs_from_dump(model_dir))                                                                                              

    # Scoring module attrs                                                                                                                                                            
    scoring_attrs_raw = scoring_module_attrs_from_dump(model_dir)                                                                                                                     
    scoring_attrs_json: dict[str, Any] | None = scoring_attrs_raw                                                                                                                     

    # Get scores summary and full scores array                                                                                                                                        
    scores_eval_summary, scores_array = scoring_eval_summary_from_pipeline(pipeline, test_texts)                                                                                      
    eval_elapsed_sec = time.perf_counter() - eval_started_at                                                                                                                          

    # Save full eval scores to JSONL for error analysis                                                                                                                               
    if scores_array is not None:                                                                                                                                                      
        qp_suffix = make_query_prompt_suffix(getattr(args, "query_prompt", None))                                                                                                     
        eval_scores_filename = f"eval_scores_{metadata['model_name']}_{metadata['mode']}_seed{metadata['seed']}{qp_suffix}.jsonl"                                                     
        eval_scores_path = (runs_dir if runs_dir is not None else get_runs_dir()) / eval_scores_filename                                                                              
        save_eval_scores(eval_scores_path, metadata, test_texts, y_true, y_pred, scores_array)                                                                                        
        print(f"Saved eval scores to: {eval_scores_path}")                                                    

    # Print results
    print()
    print("=" * 60)
    print(f"Results ({metadata['mode']})")
    print("=" * 60)
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    # Save results (extra — воспроизводимость + данные для HYP-JB-004 / порог / скоры)
    result = {
        "model_name": metadata["model_name"],
        "mode": metadata["mode"],
        "n_shots": metadata["n_shots"],
        "seed": metadata["seed"],
        "query_prompt": getattr(args, "query_prompt", None),
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "over_refusal_rate": metrics["over_refusal_rate"],
        "recall_vanilla_harmful": metrics.get("recall_vanilla_harmful"),
        "recall_adversarial_harmful": metrics.get("recall_adversarial_harmful"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extra": {
            "preset": metadata.get("preset", cli_preset),
            "search_space_preset": metadata.get(
                "autointent_preset",
                resolve_preset(metadata.get("preset", cli_preset))[1],
            ),
            "embedder": metadata["embedder"],
            "embedder_hf_model": hf_emb,
            "embedder_fixed": metadata.get("embedder_fixed", True),
            "pilot": metadata["pilot"],
            "model_dir": str(model_dir),
            "data_dir": str(data_dir),
            "results_dir": str(results_dir),
            "run_config": {
                "mode": mode,
                "n_shots": metadata.get("n_shots"),
                "seed": metadata.get("seed"),
                "preset": metadata.get("preset", "classic-light"),
                "pilot": metadata.get("pilot"),
                "no_fix_embedder": no_fix_embedder,
                "eval_only": getattr(args, "eval_only", False),
                "train_only": getattr(args, "train_only", False),
                "all_seeds": getattr(args, "all_seeds", False),
            },
            "train_metadata": metadata,
            "runtime_environment": runtime_environment_summary(),
            "timings": {
                "train_total_sec": (metadata.get("timings") or {}).get("train_total_sec"),
                "eval_total_sec": float(eval_elapsed_sec),
            },
            "data_summary": {
                "train": (metadata.get("data_summary") or {}).get("train"),
                "test": test_data_summary,
                "eval_binary": eval_binary_summary,
                "alignment": data_alignment,
            },
            "prediction_summary": {
                "n_predictions": int(len(y_pred)),
                "raw_prediction_values": dict(sorted(raw_pred_values.items())),
                "raw_none_count": raw_pred_none,
                "none_policy": "None predictions are treated as jailbreak(label=1)",
                "pred_safe": int(np.sum(y_pred == 0)),
                "pred_jailbreak": int(np.sum(y_pred == 1)),
                "by_data_type": prediction_by_type,
            },
            "eval_counts": eval_counts,
            "scoring_module_attrs": scoring_attrs_json,
            "decision_module_attrs": decision_attrs_json,
            "scores_eval_summary": scores_eval_summary,
            "artifact_manifest": pipeline_artifact_manifest(model_dir),
        },
    }

    save_metrics(result, results_dir)
    save_run_metrics_file(result, runs_dir if runs_dir is not None else get_runs_dir())

    if getattr(args, "print_hypothesis_log", False):
        hyp_payload = {
            "model_name": result["model_name"],
            "mode": result["mode"],
            "seed": result["seed"],
            "n_shots": result["n_shots"],
            "search_space_preset": (result.get("extra") or {}).get("search_space_preset"),
            "metrics_core": {
                "f1": result["f1"],
                "recall": result["recall"],
                "over_refusal_rate": result["over_refusal_rate"],
                "precision": result["precision"],
                "recall_vanilla_harmful": result.get("recall_vanilla_harmful"),
                "recall_adversarial_harmful": result.get("recall_adversarial_harmful"),
            },
            "embedder_hf_model": hf_emb,
            "eval_counts": eval_counts,
            "prediction_summary": result["extra"]["prediction_summary"],
            "data_summary": result["extra"]["data_summary"],
            "scores_eval_summary": scores_eval_summary,
            "decision_module_attrs": decision_attrs_json,
            "artifact_manifest": result["extra"]["artifact_manifest"],
            "hypothesis_notes": (
                "FNR≈fnr_jailbreak, FPR_safe≈fpr_safe; для асимметричного порога нужен "
                "sweep по threshold на val (не только эта точка)."
            ),
        }
        print()
        print("=" * 60)
        print("HYPOTHESIS_LOG (краткий JSON для отчёта / Kaggle output)")
        print("=" * 60)
        print(json.dumps(hyp_payload, indent=2, ensure_ascii=False))

    if getattr(args, "print_metrics_json", False):
        print()
        print("=" * 60)
        print("METRICS_JSON (same object as metrics.json row)")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if metadata.get("pilot") and metadata.get("embedder_fixed", True):
        print()
        print("=" * 60)
        print("PILOT run: using small embedder for fast validation")
        print("Re-run without --pilot for final results")
        print("=" * 60)


def main():
    _configure_run_logging()
    _quiet_third_party_loggers()

    parser = argparse.ArgumentParser(
        description="Run AutoIntent on WildJailbreak (train + evaluate)"
    )
    parser.add_argument(
        "--mode",
        choices=["fewshot", "full"],
        default="fewshot",
        help=(
            "fewshot: train_shot{N}_seed{S}.json; "
            "full: wildjailbreak_full100k_seed{S}.json (prepare_data --full_subset)"
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="classic-light",
        choices=list(CLI_PRESET_CHOICES),
        help=(
            "Search-space пресет AutoIntent (classic-medium, nn-medium, "
            "zero-shot-encoders, transformers-light, bert-finetune→transformers-light, …)."
        ),
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
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help=f"Прогнать подряд все сиды {list(DEFAULT_SEEDS)} (игнорирует одиночный --seed)",
    )
    parser.add_argument(
        "--no-automl-progress",
        action="store_true",
        help=(
            "Отключить tqdm Optuna и строку Overall HPO %% во время AutoML "
            "(полезно для логов без перезаписи строк)"
        ),
    )
    parser.add_argument(
        "--print-metrics-json",
        action="store_true",
        help="После каждой оценки вывести полный JSON строки метрик в stdout (удобно для Kaggle/логов)",
    )
    parser.add_argument(
        "--print-hypothesis-log",
        action="store_true",
        help=(
            "После оценки вывести компактный JSON (метрики, eval_counts, скоры, decision attrs) "
            "для отчёта по гипотезам (Kaggle output)"
        ),
    )
    parser.add_argument(
        "--query-prompt",
        type=str,
        default=None,
        help=(
            "Instruction prefix for query embeddings (E5-instruct). "
            "Applies only with fixed embedder."
        ),
    )
    args = parser.parse_args()

    seeds = list(DEFAULT_SEEDS) if args.all_seeds else [args.seed]

    data_dir = get_data_dir()
    results_dir = get_results_dir()

    for run_idx, seed in enumerate(seeds, start=1):
        args.seed = seed
        # Fix global random generators for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model_dir = get_model_dir(
            args.preset,
            args.pilot,
            args.no_fix_embedder,
            args.mode,
            args.n_shots if args.mode == "fewshot" else None,
            args.seed,
            args.query_prompt,
        )

        if len(seeds) > 1:
            print()
            print("#" * 60)
            print(f"# Seed {args.seed}  ({run_idx}/{len(seeds)})")
            print("#" * 60)

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
