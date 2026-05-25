"""
WildJailbreak — табличные AutoML-бейзлайны (H2O / AutoGluon / LightAutoML), без AutoIntent.

Kaggle: 2×T4 — задайте CUDA и `JAILBREAK_NUM_GPUS=2` (ноутбук делает это в §2b).
Тихий режим: `--metrics-only` — в stdout только блок METRICS_JSON (как строка metrics.json).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

script_dir = Path(__file__).resolve().parent
task_dir = script_dir.parent
project_root = task_dir.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SkPipeline

from tasks.jailbreak_detection.src.metrics import evaluate_jailbreak

DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456)
DEFAULT_N_SHOTS: tuple[int, ...] = (10, 20, 50)
FRAMEWORK_CHOICES = ("h2o", "autogluon", "lightautoml")


def metrics_only() -> bool:
    return os.environ.get("JAILBREAK_METRICS_ONLY", "").strip() in ("1", "true", "yes")


def _is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def _release_memory() -> None:
    gc.collect()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def num_gpus() -> int:
    v = os.environ.get("JAILBREAK_NUM_GPUS", "").strip()
    if v.isdigit():
        return max(0, int(v))
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _apply_quiet_env() -> None:
    if not metrics_only() and os.environ.get("JAILBREAK_QUIET_LOGS") != "1":
        return
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR, force=True)
    for name in ("h2o", "autogluon", "lightautoml", "transformers", "datasets", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger("lightautoml.utils.installation").setLevel(logging.ERROR)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")


def _log(msg: str) -> None:
    if not metrics_only():
        print(msg, flush=True)


def _emit_metrics_row(result: dict) -> None:
    print("METRICS_JSON", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def get_data_dir() -> Path:
    return task_dir / "data" / "processed"


def get_runs_dir() -> Path:
    d = task_dir / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_results_dir() -> Path:
    d = task_dir / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def full_train_filename(seed: int) -> str:
    return f"wildjailbreak_full100k_seed{seed}.json"


def _load_autointent_train(path: Path) -> tuple[list[str], list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    labels: list[int] = []
    for utt in data["intents"][0]["utterances"]:
        texts.append(utt)
        labels.append(0)
    for utt in data["oos_utterances"]:
        texts.append(utt)
        labels.append(1)
    return texts, labels


def load_train(mode: str, seed: int, n_shots: int | None, data_dir: Path) -> pd.DataFrame:
    if mode == "full":
        path = data_dir / full_train_filename(seed)
    else:
        assert n_shots is not None
        path = data_dir / f"train_shot{n_shots}_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    texts, labels = _load_autointent_train(path)
    return pd.DataFrame({"text": texts, "label": labels})


def load_eval(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test = json.loads((data_dir / "test.json").read_text(encoding="utf-8"))
    eval_rows = []
    with open(data_dir / "wildjailbreak_eval_binary.jsonl", encoding="utf-8") as f:
        for line in f:
            eval_rows.append(json.loads(line))
    if len(test["utterances"]) != len(eval_rows):
        raise ValueError("test.json and eval_binary length mismatch")
    df = pd.DataFrame({"text": test["utterances"]})
    y_true = np.array([1 if r["binary_label"] == "jailbreak" else 0 for r in eval_rows])
    data_types = np.array([r["data_type"] for r in eval_rows])
    return df, y_true, data_types


def confusion_and_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp != 1)))
    tn = int(np.sum((yt == 0) & (yp != 1)))
    denom_p = tp + fn
    denom_n = fp + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fnr_jailbreak": float(fn / denom_p) if denom_p else None,
        "fpr_safe": float(fp / denom_n) if denom_n else None,
        "n_eval": int(len(yt)),
        "n_safe_true": int(np.sum(yt == 0)),
        "n_jailbreak_true": int(np.sum(yt == 1)),
    }


def proba_summary(y_proba: np.ndarray | None) -> dict[str, Any] | None:
    if y_proba is None:
        return None
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    margin = 2.0 * p - 1.0
    return {
        "n_scored": int(len(p)),
        "n_score_dims": 1,
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
        "margin_min": float(np.min(margin)),
        "margin_max": float(np.max(margin)),
        "score_col0_mean": float(np.mean(1.0 - p)),
        "score_col1_mean": float(np.mean(p)),
        "note": "P(jailbreak); margin=2*p-1",
    }


def model_name(framework: str) -> str:
    return f"{framework}_automl_baseline"


def model_dir_path(framework: str, mode: str, seed: int, n_shots: int | None) -> Path:
    mn = model_name(framework)
    if mode == "full":
        return get_runs_dir() / f"{mn}_full_seed{seed}"
    return get_runs_dir() / f"{mn}_{n_shots}shot_seed{seed}"


def default_time_limit(mode: str, framework: str) -> int:
    if framework == "h2o":
        # H2O+Java тяжёлый: короче лимит, иначе OOM/обрыв сессии Kaggle
        return 2400 if mode == "full" else 900
    return 3600 if mode == "full" else 900


def _h2o_max_mem() -> str:
    if os.environ.get("H2O_MAX_MEM"):
        return os.environ["H2O_MAX_MEM"]
    # Kaggle GPU ~30GB RAM: оставляем запас под Python/AG/LAMA в ядре ноутбука
    return "14G" if _is_kaggle() else "18G"


def _h2o_init() -> None:
    import h2o

    if h2o.connection():
        return
    h2o.init(verbose=False, max_mem_size=_h2o_max_mem(), nthreads=-1)


def _h2o_shutdown() -> None:
    try:
        import h2o

        if h2o.connection():
            h2o.cluster().shutdown(prompt=False)
        h2o.disconnect()
    except Exception:
        pass


def embedder_label(framework: str, ngpu: int) -> str:
    if framework == "lightautoml":
        base = "tfidf_svd_sklearn"
    elif framework == "autogluon":
        base = "autogluon_text_default"
    else:
        base = "tfidf_svd_h2o"
    if ngpu > 0:
        return f"{base}_gpu{ngpu}"
    return base


def build_metrics_row(
    framework: str,
    meta_mode: str,
    seed: int,
    n_shots: int | None,
    mdir: Path,
    metrics: dict,
    eval_counts: dict,
    scores_eval_summary: dict | None,
    ngpu: int,
) -> dict:
    return {
        "model_name": model_name(framework),
        "mode": meta_mode,
        "n_shots": n_shots,
        "seed": seed,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "over_refusal_rate": metrics["over_refusal_rate"],
        "recall_adversarial_harmful": metrics.get("recall_adversarial_harmful"),
        "recall_vanilla_harmful": metrics.get("recall_vanilla_harmful"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extra": {
            "preset": f"{framework}-default",
            "embedder": embedder_label(framework, ngpu),
            "embedder_hf_model": None,
            "embedder_fixed": True,
            "pilot": False,
            "model_dir": str(mdir),
            "eval_counts": eval_counts,
            "decision_module_attrs": None,
            "scores_eval_summary": scores_eval_summary,
        },
    }


def _mirror_metrics(metrics_path: Path) -> None:
    kw = Path("/kaggle/working")
    if kw.exists() and metrics_path.is_file():
        shutil.copy2(metrics_path, kw / "metrics_jailbreak_automl_latest.json")


def save_to_files() -> bool:
    return os.environ.get("JAILBREAK_METRICS_OUTPUT_FILES", "1").strip() not in (
        "0",
        "false",
        "no",
    )


def save_metrics(result: dict) -> None:
    if not save_to_files():
        return
    metrics_path = get_results_dir() / "metrics.json"
    rows: list[dict] = []
    if metrics_path.exists():
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            rows = []
    rows = [
        r
        for r in rows
        if not (
            r.get("model_name") == result["model_name"]
            and r.get("mode") == result["mode"]
            and r.get("n_shots") == result.get("n_shots")
            and r.get("seed") == result.get("seed")
        )
    ]
    rows.append(result)
    metrics_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _mirror_metrics(metrics_path)


def train_h2o(
    train_df: pd.DataFrame,
    model_dir: Path,
    *,
    mode: str,
    seed: int,
    time_limit_sec: int,
    ngpu: int,
) -> tuple[Any, float, SkPipeline]:
    import h2o
    from h2o.automl import H2OAutoML

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tab_train, pipe = train_df_tfidf(train_df, model_dir, mode=mode, seed=seed)
    feat_cols = [c for c in tab_train.columns if c.startswith("f")]
    _release_memory()

    if ngpu > 0:
        os.environ["H2O_XGBOOST_GPU"] = "1"

    _h2o_init()
    t0 = time.perf_counter()
    hf = h2o.H2OFrame(tab_train)
    del tab_train
    _release_memory()
    hf["label"] = hf["label"].asfactor()

    aml_kwargs: dict[str, Any] = {
        "max_runtime_secs": time_limit_sec,
        "max_models": int(os.environ.get("H2O_MAX_MODELS", "16")),
        "max_runtime_secs_per_model": int(os.environ.get("H2O_MAX_SEC_PER_MODEL", "600")),
        "seed": seed,
        "sort_metric": "F1",
        "balance_classes": True,
        "project_name": f"jb_{model_dir.name}"[:64],
    }
    # H2O: только exclude_algos ИЛИ include_algos, не оба
    if ngpu > 0:
        aml_kwargs["include_algos"] = ["GBM", "XGBoost", "GLM", "DRF"]
    else:
        aml_kwargs["exclude_algos"] = ["DeepLearning"]
    aml = H2OAutoML(**aml_kwargs)
    aml.train(y="label", training_frame=hf, x=feat_cols)
    del hf
    _release_memory()
    train_sec = time.perf_counter() - t0

    if aml.leader is None:
        raise RuntimeError("H2O AutoML finished without a leader model")

    (model_dir / "h2o_leader.txt").write_text(str(aml.leader), encoding="utf-8")
    try:
        h2o.save_model(model=aml.leader, path=str(model_dir), force=True)
    except Exception:
        pass
    return aml, train_sec, pipe


def predict_h2o(
    model_or_aml: Any,
    test_df: pd.DataFrame,
    pipe: SkPipeline,
) -> tuple[np.ndarray, np.ndarray | None]:
    import h2o

    _h2o_init()
    leader = model_or_aml.leader if hasattr(model_or_aml, "leader") else model_or_aml
    tab_test = test_df_tfidf(test_df, pipe)
    ht = h2o.H2OFrame(tab_test)
    pred = leader.predict(ht)
    pdf = pred.as_data_frame(use_pandas=True)
    col = "predict" if "predict" in pdf.columns else pdf.columns[0]
    labels = (
        pdf[col]
        .astype(str)
        .map({"0": 0, "1": 1, "0.0": 0, "1.0": 1})
        .fillna(1)
        .astype(int)
        .to_numpy()
    )
    proba = pdf["p1"].to_numpy(dtype=np.float64) if "p1" in pdf.columns else None
    del ht, tab_test, pdf, pred
    _release_memory()
    return labels, proba


def train_autogluon(
    train_df: pd.DataFrame,
    model_dir: Path,
    *,
    mode: str,
    time_limit_sec: int,
    ngpu: int,
) -> tuple[Any, float]:
    from autogluon.tabular import TabularPredictor

    if model_dir.exists():
        shutil.rmtree(model_dir)
    t0 = time.perf_counter()
    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        eval_metric="f1",
        path=str(model_dir),
        verbosity=0,
    )
    ag_fit: dict[str, Any] = {"num_cpus": max(1, (os.cpu_count() or 4) - 2)}
    if ngpu > 0:
        # 1 GPU за раз — ниже пик VRAM при full+text
        ag_fit["num_gpus"] = min(ngpu, _env_int("AG_NUM_GPUS", 1))
    excluded = ["NN_TORCH", "FASTAI", "KNN"]
    if mode == "full":
        excluded.append("RF")
    predictor.fit(
        train_df,
        time_limit=time_limit_sec,
        presets=None,
        excluded_model_types=excluded,
        ag_args_fit=ag_fit,
    )
    _release_memory()
    return predictor, time.perf_counter() - t0


def predict_autogluon(predictor: Any, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = predictor.predict(test_df).to_numpy().astype(int)
    proba = None
    try:
        proba = predictor.predict_proba(test_df)[1].to_numpy(dtype=np.float64)
    except Exception:
        pass
    return y_pred, proba


def _tfidf_dims(mode: str) -> tuple[int, int]:
    if mode == "full":
        return (
            _env_int("JAILBREAK_TFIDF_MAX_FEATURES_FULL", 4000),
            _env_int("JAILBREAK_TFIDF_SVD_FULL", 128),
        )
    return (
        _env_int("JAILBREAK_TFIDF_MAX_FEATURES_FEWSHOT", 4000),
        _env_int("JAILBREAK_TFIDF_SVD_FEWSHOT", 128),
    )


def _tfidf_svd_pipeline(mode: str, seed: int) -> SkPipeline:
    max_feat, n_comp = _tfidf_dims(mode)
    return SkPipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_feat,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            ("svd", TruncatedSVD(n_components=n_comp, random_state=seed)),
        ]
    )


def train_df_tfidf(
    train_df: pd.DataFrame,
    model_dir: Path,
    *,
    mode: str,
    seed: int,
) -> tuple[pd.DataFrame, SkPipeline]:
    import joblib

    pipe = _tfidf_svd_pipeline(mode, seed)
    feats = pipe.fit_transform(train_df["text"].astype(str))
    arr = np.asarray(feats, dtype=np.float32)
    del feats
    _release_memory()
    out = pd.DataFrame({f"f{i}": arr[:, i] for i in range(arr.shape[1])})
    out["label"] = train_df["label"].astype(int).values
    del arr
    joblib.dump(pipe, model_dir / "tfidf_svd.joblib")
    return out, pipe


def test_df_tfidf(test_df: pd.DataFrame, pipe: SkPipeline) -> pd.DataFrame:
    feats = pipe.transform(test_df["text"].astype(str))
    arr = np.asarray(feats, dtype=np.float32)
    del feats
    out = pd.DataFrame({f"f{i}": arr[:, i] for i in range(arr.shape[1])})
    del arr
    return out


def _lama_gpu_ids(ngpu: int) -> str:
    custom = os.environ.get("JAILBREAK_LAMA_GPU_IDS", "").strip()
    if custom:
        return custom
    n = min(ngpu, _env_int("AG_NUM_GPUS", 1)) if ngpu > 0 else 0
    return ",".join(str(i) for i in range(max(1, n)))


def train_lightautoml(
    train_df: pd.DataFrame,
    model_dir: Path,
    *,
    mode: str,
    seed: int,
    time_limit_sec: int,
    ngpu: int,
) -> tuple[Any, float, SkPipeline]:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tab_train, pipe = train_df_tfidf(train_df, model_dir, mode=mode, seed=seed)
    feat_cols = [c for c in tab_train.columns if c.startswith("f")]
    tab_train["label"] = tab_train["label"].astype(np.int8)
    # LAMA 0.4: roles = {role_name: column_or_list}, не {column: role}
    roles = {
        "target": "label",
        "numeric": feat_cols,
    }

    kwargs: dict[str, Any] = {
        "task": Task("binary"),
        "timeout": time_limit_sec,
        "cpu_limit": max(1, (os.cpu_count() or 4) - 2),
        "memory_limit": _env_int("JAILBREAK_LAMA_MEMORY_LIMIT", 12),
        "reader_params": {
            "random_state": seed,
            "cv": _env_int("JAILBREAK_LAMA_CV", 3),
            "n_jobs": 1,
        },
    }
    if ngpu > 0:
        kwargs["gpu_ids"] = _lama_gpu_ids(ngpu)
    else:
        kwargs["gpu_ids"] = None

    automl = TabularAutoML(**kwargs)
    t0 = time.perf_counter()
    automl.fit_predict(train_data=tab_train, roles=roles, verbose=0)
    del tab_train
    _release_memory()
    return automl, time.perf_counter() - t0, pipe


def predict_lightautoml(
    automl: Any,
    test_df: pd.DataFrame,
    pipe: SkPipeline,
) -> tuple[np.ndarray, np.ndarray | None]:
    tab_test = test_df_tfidf(test_df, pipe)
    pred = automl.predict(tab_test)
    arr = np.asarray(pred.data if hasattr(pred, "data") else pred, dtype=np.float64)
    # LAMA binary: shape (n, 1) = P(label=1). argmax на 1 колонке всегда 0 → F1=0.
    if arr.ndim == 1:
        proba = arr.ravel()
    elif arr.shape[1] == 1:
        proba = arr[:, 0]
    else:
        proba = arr[:, 1] if arr.shape[1] > 1 else arr[:, 0]
    y_pred = (proba > 0.5).astype(int)
    return y_pred, proba


def run_one(
    framework: str,
    mode: str,
    seed: int,
    n_shots: int | None,
    *,
    time_limit_sec: int,
    metrics_only_mode: bool,
    train_only: bool,
    eval_only: bool,
) -> dict:
    _apply_quiet_env()
    ngpu = num_gpus()
    data_dir = get_data_dir()
    meta_mode = "full" if mode == "full" else f"{n_shots}shot"
    mdir = model_dir_path(framework, mode, seed, n_shots)

    _log(f"{framework} {meta_mode} seed={seed} gpus={ngpu}")

    train_df = load_train(mode, seed, n_shots, data_dir)
    test_df, y_true, data_types = load_eval(data_dir)

    artifact: Any = None
    feat_pipe: SkPipeline | None = None

    try:
        if not eval_only:
            if framework == "h2o":
                artifact, _, feat_pipe = train_h2o(
                    train_df,
                    mdir,
                    mode=mode,
                    seed=seed,
                    time_limit_sec=time_limit_sec,
                    ngpu=ngpu,
                )
            elif framework == "autogluon":
                artifact, _ = train_autogluon(
                    train_df,
                    mdir,
                    mode=mode,
                    time_limit_sec=time_limit_sec,
                    ngpu=ngpu,
                )
            elif framework == "lightautoml":
                artifact, _, feat_pipe = train_lightautoml(
                    train_df,
                    mdir,
                    mode=mode,
                    seed=seed,
                    time_limit_sec=time_limit_sec,
                    ngpu=ngpu,
                )
            else:
                raise ValueError(framework)
            del train_df
            _release_memory()

        if train_only:
            return {}

        if eval_only and artifact is None:
            if framework == "h2o":
                import h2o

                _h2o_init()
                artifact = h2o.load_model(str(mdir))
            elif framework == "autogluon":
                from autogluon.tabular import TabularPredictor

                artifact = TabularPredictor.load(str(mdir))
            elif framework == "lightautoml":
                raise NotImplementedError("lightautoml eval-only не поддержан")

        if framework in ("h2o", "lightautoml") and feat_pipe is None:
            import joblib

            feat_pipe = joblib.load(mdir / "tfidf_svd.joblib")

        if framework == "h2o":
            y_pred, y_proba = predict_h2o(artifact, test_df, feat_pipe)
        elif framework == "autogluon":
            y_pred, y_proba = predict_autogluon(artifact, test_df)
        else:
            y_pred, y_proba = predict_lightautoml(artifact, test_df, feat_pipe)

        metrics = evaluate_jailbreak(y_true, y_pred, data_types, oos_label=1)
        result = build_metrics_row(
            framework,
            meta_mode,
            seed,
            n_shots,
            mdir,
            metrics,
            confusion_and_rates(y_true, y_pred),
            proba_summary(y_proba),
            ngpu,
        )

        if metrics_only_mode:
            _emit_metrics_row(result)
        else:
            _log(json.dumps(metrics, ensure_ascii=False))

        save_metrics(result)
        return result
    finally:
        if framework == "h2o":
            _h2o_shutdown()
        _release_memory()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular AutoML baselines on WildJailbreak")
    parser.add_argument("--framework", choices=FRAMEWORK_CHOICES, required=True)
    parser.add_argument("--mode", choices=["fewshot", "full"], default="fewshot")
    parser.add_argument("--n_shots", type=int, choices=[10, 20, 50], default=10)
    parser.add_argument("--seed", type=int, choices=[42, 123, 456], default=42)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--all-shots", action="store_true")
    parser.add_argument("--time-limit-sec", type=int, default=None)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Только METRICS_JSON в stdout (формат metrics.json)",
    )
    parser.add_argument(
        "--print-metrics-json",
        action="store_true",
        help="Устар.: то же, что --metrics-only",
    )
    parser.add_argument(
        "--no-save-files",
        action="store_true",
        help="Не писать metrics.json (только stdout METRICS_JSON)",
    )
    args = parser.parse_args()

    mom = args.metrics_only or args.print_metrics_json or metrics_only()
    if args.no_save_files:
        os.environ["JAILBREAK_METRICS_OUTPUT_FILES"] = "0"
    if mom:
        os.environ["JAILBREAK_METRICS_ONLY"] = "1"

    if args.mode == "full" and args.all_shots:
        parser.error("--all-shots only for fewshot")

    seeds = list(DEFAULT_SEEDS) if args.all_seeds else [args.seed]
    shots = list(DEFAULT_N_SHOTS) if args.all_shots and args.mode == "fewshot" else [args.n_shots]
    tlim = args.time_limit_sec or default_time_limit(args.mode, args.framework)

    for seed in seeds:
        for n_shots in ([None] if args.mode == "full" else shots):
            run_one(
                args.framework,
                args.mode,
                seed,
                n_shots,
                time_limit_sec=tlim,
                metrics_only_mode=mom,
                train_only=args.train_only,
                eval_only=args.eval_only,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
