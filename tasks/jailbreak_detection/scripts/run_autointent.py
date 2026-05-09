"""
Запуск AutoIntent на WildJailbreak (обучение + оценка).

Использование:
    # Pilot (маленький embedder, быстрая валидация)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --pilot

    # Final (оригинальный embedder e5)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42

    # AutoML без фиксации embedder (медленнее, потенциально лучше)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --no-fix-embedder

    # Только оценка (модель должна существовать)
    python scripts/run_autointent.py --mode fewshot --n_shots 10 --seed 42 --eval-only

    # Full train (100K stratified wildjailbreak_full100k_seed{S}.json из prepare_data --full_subset)
    python scripts/run_autointent.py --mode full --seed 42

    # Full на всех сидах (42, 123, 456) подряд
    python scripts/run_autointent.py --mode full --all-seeds

Данные:
    - Few-shot train: data/processed/train_shot{N}_seed{S}.json
    - Full train: data/processed/wildjailbreak_full100k_seed{S}.json
      (это 100K стратифицированная подвыборка, не весь сырой WildJailbreak train;
      готовится скриптом prepare_data.py --full_subset из‑за размера и времени AutoML)
    - Test: data/processed/test.json (для inference)
    - Eval binary: data/processed/wildjailbreak_eval_binary.jsonl (для метрик)

Результаты:
    - Модель few-shot (фикс. e5): runs/autointent_classic-light_{N}shot_seed{S}/
    - Модель few-shot (автоэмбеддер): runs/autointent_classic-light_autoembedder_{N}shot_seed{S}/
    - Модель full: runs/autointent_classic-light_full_seed{S}/ (и аналоги для pilot / autoembedder)
    - Метрики: results/metrics.json (числовые поля + extra: embedder, eval_counts TP/FP/FN/TN,
      scores_eval_summary, decision_module_attrs, model_dir и т.д.)
    - На каждый прогон оценки: runs/metrics_<model_name>_<mode>_seed<S>.json — тот же JSON-объект, что одна строка в metrics.json
    - Дублирование метрик в stdout: --print-metrics-json (Kaggle/логи, если файлы не сохранились)

    Пример: few-shot без фиксации эмбеддера на всех сидах:
        python scripts/run_autointent.py --mode fewshot --n_shots 10 --no-fix-embedder --all-seeds

Стабильность на macOS ARM:
    load_dotenv и лимиты потоков BLAS/OpenMP должны применяться до import numpy/torch.
    Скопируйте .env.example в .env при segfault/sklearn warnings во время AutoML.

Прогресс AutoML (по умолчанию):
    полоса Optuna (tqdm) по числу trials в текущем узле (scoring/decision);
    строка «Overall HPO (estimate)» — грубая доля по этапам HPO (не покрывает долгий первый embed внутри trial).
    Отключить: --no-automl-progress
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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


_apply_thread_env_defaults()

import numpy as np

sys.path.insert(0, str(project_root))

from autointent import Pipeline, Dataset as AIDataset
from autointent.configs import LoggingConfig, EmbedderConfig, DataConfig

from tasks.jailbreak_detection.src.metrics import evaluate_jailbreak

logger = logging.getLogger(__name__)


def _configure_run_logging() -> None:
    """Один раз настраивает stderr-logging для сообщений об эмбеддере и др."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )


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
) -> dict[str, Any] | None:
    """
    Краткая статистика по скорам scoring (margin между классами 0 и 1).
    Предполагается порядок классов как в train: 0=safe, 1=jailbreak.
    """
    try:
        inf_out = pipeline.predict_with_metadata(texts)
        utterances = getattr(inf_out, "utterances", None)
        if not isinstance(utterances, list) or not utterances:
            return None
        rows = []
        for u in utterances:
            sc = getattr(u, "score", None)
            if sc is None:
                return None
            rows.append(np.asarray(sc, dtype=np.float64).ravel())
        scores = np.stack(rows, axis=0)
    except Exception as exc:
        logger.debug("scoring_eval_summary_from_pipeline: %s", exc)
        return None
    if scores.ndim != 2 or scores.shape[1] < 2:
        return {
            "note": "unexpected_score_shape",
            "shape": list(scores.shape),
        }
    margin = scores[:, 1] - scores[:, 0]
    return {
        "n_scored": int(scores.shape[0]),
        "n_score_dims": int(scores.shape[1]),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
        "margin_min": float(np.min(margin)),
        "margin_max": float(np.max(margin)),
        "score_col0_mean": float(np.mean(scores[:, 0])),
        "score_col1_mean": float(np.mean(scores[:, 1])),
        "note": "col0≈safe intent col1≈jailbreak (classic-light binary setup)",
    }


def get_model_name(pilot: bool, no_fix_embedder: bool) -> str:
    if no_fix_embedder:
        return "autointent_classic-light_autoembedder"
    if pilot:
        return "autointent_classic-light_pilot"
    return "autointent_classic-light"


def get_embedder_name(pilot: bool) -> str:
    if pilot:
        return "intfloat/multilingual-e5-small"
    return "intfloat/multilingual-e5-large-instruct"


def get_model_dir(
    pilot: bool,
    no_fix_embedder: bool,
    mode: str,
    n_shots: int | None,
    seed: int,
) -> Path:
    model_name = get_model_name(pilot, no_fix_embedder)
    if mode == "full":
        return get_runs_dir() / f"{model_name}_full_seed{seed}"
    assert n_shots is not None
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


def run_metrics_filename(result: dict) -> str:
    """Stable filename for per-run metrics JSON under runs/ (one object = one metrics.json row)."""
    mn = result["model_name"].replace("/", "_").replace(" ", "_")
    mode = result["mode"]
    seed = result["seed"]
    if mode == "full":
        return f"metrics_{mn}_full_seed{seed}.json"
    return f"metrics_{mn}_{mode}_seed{seed}.json"


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
    print(f"Embedder: {embedder_name}")
    if no_fix_embedder:
        logger.info(
            "Embedder policy: AutoML (preset classic-light); "
            "конкретный HuggingFace id будет известен после dump в конце обучения.",
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
        train_raw = load_full_train(args.seed, data_dir)
    else:
        train_raw = load_fewshot_train(args.n_shots, args.seed, data_dir)
    test_raw = load_test(data_dir)

    train_ai = convert_to_autointent_train(train_raw)
    test_ai = convert_to_autointent_test(test_raw)

    # Binary classification: 2 intents (no OOS)
    intents = [
        {"id": 0, "name": "safe"},
        {"id": 1, "name": "jailbreak"},
    ]

    n_safe = len(train_raw["intents"][0]["utterances"])
    n_jailbreak = len(train_raw["oos_utterances"])
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
    if not no_fix_embedder:
        pipeline.set_config(EmbedderConfig(model_name=embedder_name))

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
        "model_name": get_model_name(args.pilot, no_fix_embedder),
        "mode": meta_mode,
        "n_shots": meta_n_shots,
        "seed": args.seed,
        "embedder": embedder_name,
        "embedder_fixed": not no_fix_embedder,
        "pilot": args.pilot,
        "preset": "classic-light",
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
    print()
    print("=" * 60)
    print("AutoIntent Evaluation (Jailbreak Detection)")
    print("=" * 60)

    # Load metadata
    metadata_path = model_dir / "train_metadata.json"
    no_fix_embedder = getattr(args, "no_fix_embedder", False)
    mode = getattr(args, "mode", "fewshot")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        if mode == "full":
            meta_mode = "full"
            meta_n_shots = None
        else:
            meta_mode = f"{args.n_shots}shot"
            meta_n_shots = args.n_shots
        metadata = {
            "model_name": get_model_name(args.pilot, no_fix_embedder),
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

    eval_counts = confusion_and_rates_jailbreak_positive(y_true, y_pred, positive_label=1)
    decision_attrs_raw = decision_module_attrs_from_dump(model_dir)
    decision_attrs_json: dict[str, Any] | None
    if decision_attrs_raw is None:
        decision_attrs_json = None
    else:
        try:
            dumped = json.dumps(decision_attrs_raw)
            if len(dumped) > 12_000:
                decision_attrs_json = {
                    "_truncated": True,
                    "top_level_keys": list(decision_attrs_raw.keys()),
                }
            else:
                decision_attrs_json = decision_attrs_raw
        except (TypeError, ValueError):
            decision_attrs_json = {"_error": "non_serializable_decision_attrs"}

    scores_eval_summary = scoring_eval_summary_from_pipeline(pipeline, test_texts)

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
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "over_refusal_rate": metrics["over_refusal_rate"],
        "recall_adversarial_harmful": metrics.get("recall_adversarial_harmful"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extra": {
            "preset": metadata.get("preset", "classic-light"),
            "embedder": metadata["embedder"],
            "embedder_hf_model": hf_emb,
            "embedder_fixed": metadata.get("embedder_fixed", True),
            "pilot": metadata["pilot"],
            "model_dir": str(model_dir),
            "eval_counts": eval_counts,
            "decision_module_attrs": decision_attrs_json,
            "scores_eval_summary": scores_eval_summary,
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
            "metrics_core": {
                "f1": result["f1"],
                "recall": result["recall"],
                "over_refusal_rate": result["over_refusal_rate"],
                "precision": result["precision"],
            },
            "embedder_hf_model": hf_emb,
            "eval_counts": eval_counts,
            "scores_eval_summary": scores_eval_summary,
            "decision_module_attrs": decision_attrs_json,
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
    args = parser.parse_args()

    seeds = list(DEFAULT_SEEDS) if args.all_seeds else [args.seed]

    data_dir = get_data_dir()
    results_dir = get_results_dir()

    for run_idx, seed in enumerate(seeds, start=1):
        args.seed = seed
        model_dir = get_model_dir(
            args.pilot,
            args.no_fix_embedder,
            args.mode,
            args.n_shots if args.mode == "fewshot" else None,
            args.seed,
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
