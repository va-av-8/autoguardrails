#!/usr/bin/env python3
"""Build self-contained Kaggle notebook: bundled repo code, pending argmax jobs only."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
OUT = TASK_DIR / "notebooks" / "kaggle_oos_automl_frameworks.ipynb"
METRICS_PATH = TASK_DIR / "results" / "metrics.json"

SEEDS = [42, 123, 456]
N_SHOTS_LIST = [10, 20, 50]
FRAMEWORKS = ("h2o", "lama", "autogluon")
MODEL_BY_FW = {"h2o": "h2o_argmax", "lama": "lama_argmax", "autogluon": "autogluon_argmax"}

BUNDLE_PATHS = [
    "tasks/__init__.py",
    "tasks/oos_detection/__init__.py",
    "tasks/oos_detection/src/__init__.py",
    "tasks/oos_detection/src/metrics.py",
    "tasks/oos_detection/src/evaluation.py",
    "tasks/oos_detection/src/data_utils.py",
    "tasks/oos_detection/src/experiment_runner.py",
    "tasks/oos_detection/src/framework_wrappers/__init__.py",
    "tasks/oos_detection/src/framework_wrappers/base.py",
    "tasks/oos_detection/src/framework_wrappers/registry.py",
    "tasks/oos_detection/src/framework_wrappers/autogluon_wrapper.py",
    "tasks/oos_detection/src/framework_wrappers/h2o_wrapper.py",
    "tasks/oos_detection/src/framework_wrappers/lama_wrapper.py",
    "tasks/oos_detection/scripts/__init__.py",
    "tasks/oos_detection/scripts/prepare_data.py",
]

FILE_MAP = {
    "tasks/oos_detection/src/metrics.py": TASK_DIR / "src/metrics.py",
    "tasks/oos_detection/src/evaluation.py": TASK_DIR / "src/evaluation.py",
    "tasks/oos_detection/src/data_utils.py": TASK_DIR / "src/data_utils.py",
    "tasks/oos_detection/src/experiment_runner.py": TASK_DIR / "src/experiment_runner.py",
    "tasks/oos_detection/src/framework_wrappers/__init__.py": TASK_DIR / "src/framework_wrappers/__init__.py",
    "tasks/oos_detection/src/framework_wrappers/base.py": TASK_DIR / "src/framework_wrappers/base.py",
    "tasks/oos_detection/src/framework_wrappers/registry.py": TASK_DIR / "src/framework_wrappers/registry.py",
    "tasks/oos_detection/src/framework_wrappers/autogluon_wrapper.py": TASK_DIR / "src/framework_wrappers/autogluon_wrapper.py",
    "tasks/oos_detection/src/framework_wrappers/h2o_wrapper.py": TASK_DIR / "src/framework_wrappers/h2o_wrapper.py",
    "tasks/oos_detection/src/framework_wrappers/lama_wrapper.py": TASK_DIR / "src/framework_wrappers/lama_wrapper.py",
    "tasks/oos_detection/scripts/prepare_data.py": TASK_DIR / "scripts/prepare_data.py",
}


def is_degenerate(rec: dict) -> bool:
    ind = rec.get("in_domain_acc")
    oos_r = rec.get("oos_recall")
    auroc = rec.get("auroc")
    if ind is not None and oos_r is not None and ind < 0.5 and oos_r > 0.9:
        return True
    if auroc is not None and auroc <= 0.55 and oos_r is not None and oos_r > 0.9:
        return True
    return False


def is_done(rec: dict) -> bool:
    if (rec.get("extra") or {}).get("prediction_mode") != "argmax":
        return False
    if rec.get("f1_oos") is None or is_degenerate(rec):
        return False
    return rec.get("model_name") in MODEL_BY_FW.values()


def _read(rel: str) -> str:
    if rel.endswith("__init__.py") and rel not in FILE_MAP:
        return ""
    path = FILE_MAP[rel]
    content = path.read_text(encoding="utf-8")
    if rel.endswith("prepare_data.py"):
        old = '    ds = load_dataset("DeepPavlov/clinc150")'
        new = (
            '    import os\n'
            '    _tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")\n'
            '    _kw = {"token": _tok} if _tok else {}\n'
            '    ds = load_dataset("DeepPavlov/clinc150", **_kw)'
        )
        content = content.replace(old, new)
    return content


def _collect_bundle() -> dict[str, str]:
    return {rel: _read(rel) for rel in BUNDLE_PATHS}


def _all_jobs() -> list[dict]:
    jobs: list[dict] = []
    for fw in FRAMEWORKS:
        for seed in SEEDS:
            jobs.append(
                {"framework": fw, "mode": "full", "n_shots": None, "seed": seed, "mode_str": "full"}
            )
        for n in N_SHOTS_LIST:
            for seed in SEEDS:
                jobs.append(
                    {
                        "framework": fw,
                        "mode": "fewshot",
                        "n_shots": n,
                        "seed": seed,
                        "mode_str": f"{n}shot",
                    }
                )
    return jobs


def _done_keys(metrics_path: Path) -> set[tuple[str, str, int]]:
    if not metrics_path.exists():
        return set()
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    done: set[tuple[str, str, int]] = set()
    for r in rows:
        if not is_done(r):
            continue
        done.add((r["model_name"], r["mode"], r["seed"]))
    return done


def compute_pending_jobs(metrics_path: Path = METRICS_PATH) -> list[dict]:
    done = _done_keys(metrics_path)
    pending: list[dict] = []
    for job in _all_jobs():
        key = (MODEL_BY_FW[job["framework"]], job["mode_str"], job["seed"])
        if key not in done:
            pending.append(job)
    return pending


def cell_md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


def cell_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")],
    }


def main() -> None:
    pending = compute_pending_jobs()
    pending_json = json.dumps(pending, ensure_ascii=False)
    bundle = _collect_bundle()
    bundle_json = json.dumps(bundle, ensure_ascii=False)

    by_fw: dict[str, int] = {}
    for j in pending:
        by_fw[j["framework"]] = by_fw.get(j["framework"], 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_fw.items())) or "нет"

    md = f"""# OOS — AutoML argmax (недостающие прогоны)

Код из репозитория **вшит в ноутбук** (без git). Стек: `prepare_data` → `experiment_runner` → AutoGluon / H2O / LAMA.

**Сейчас в очереди: {len(pending)} jobs** ({summary})

- AutoGluon: 12/12 уже в `metrics.json` — **не запускается**
- Пропуск успешных прогонов: встроенный список + файл `automl_frameworks_argmax_metrics.json` на Kaggle
- Логи: только `RUN_START` / `RUN_FINISH`
- Укажите **HF_TOKEN** в ячейке 2 (или Kaggle Secrets)
- GPU **T4**, Internet On"""

    setup = r'''# 1. Setup
import os, sys, subprocess, json, logging, warnings
from pathlib import Path

for _k, _v in {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error",
    "OOS_METRICS_LOG": "compact", "OOS_QUIET_FIT": "1", "OOS_H2O_QUIET": "1",
    "H2O_MAX_MEM": "6G", "H2O_NTHREADS": "2",
}.items():
    os.environ.setdefault(_k, _v)

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "datasets>=2.14", "sentence-transformers>=2.7", "scikit-learn>=1.3",
    "pandas", "h2o>=3.44", "autogluon.tabular>=1.1", "lightautoml>=0.3.8"],
    check=False)

WORKDIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
WORKDIR.mkdir(parents=True, exist_ok=True)'''

    hf = r'''# 2. HuggingFace token + paths
HF_TOKEN = ""  # <-- вставьте токен или оставьте пустым для Kaggle Secrets

if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        for _name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            try:
                HF_TOKEN = UserSecretsClient().get_secret(_name)
                if HF_TOKEN:
                    break
            except Exception:
                pass
    except Exception:
        pass

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass

BUNDLE_ROOT = WORKDIR / "oos_bundle"
METRICS_JSON = WORKDIR / "automl_frameworks_argmax_metrics.json"
SOURCE = "deeppavlov"
EMBEDDER = "intfloat/multilingual-e5-large-instruct"'''

    install_bundle = (
        "# 3. Install bundled repo code (same files as tasks/oos_detection/)\n"
        "import json\n"
        f"BUNDLE_FILES = json.loads({bundle_json!r})\n"
        "for rel, content in BUNDLE_FILES.items():\n"
        "    p = BUNDLE_ROOT / rel\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_text(content, encoding='utf-8')\n"
        "sys.path.insert(0, str(BUNDLE_ROOT))\n"
    )

    prepare = r'''# 4. Prepare data (prepare_data.py — deeppavlov)
import contextlib
import io

from tasks.oos_detection.scripts.prepare_data import prepare_source

_proc = BUNDLE_ROOT / "tasks" / "oos_detection" / "data" / "processed" / SOURCE
if not (_proc / "full.json").exists():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        prepare_source(SOURCE)'''

    run = textwrap.dedent(
        f'''\
        # 5. Pending argmax grid ({len(pending)} jobs — built from local metrics.json)
        import json
        import logging
        from pathlib import Path

        from tasks.oos_detection.src.experiment_runner import is_degenerate_result, run_single_experiment

        logging.disable(logging.CRITICAL)

        MODEL_BY_FW = {json.dumps(MODEL_BY_FW)}
        PENDING_JOBS = json.loads({pending_json!r})

        def _wrapper_kwargs(framework, mode, n_shots):
            kw = {{"embedder_name": EMBEDDER, "prediction_mode": "argmax"}}
            if framework == "h2o":
                kw["max_models"] = 5
                kw["max_runtime_secs"] = 900 if mode == "full" else 600
            elif framework == "lama":
                kw["cpu_limit"] = 1
                if mode == "full":
                    kw["timeout"] = 2100
                elif n_shots == 10:
                    kw["timeout"] = 900
                elif n_shots == 20:
                    kw["timeout"] = 1200
                elif n_shots == 50:
                    kw["timeout"] = 1500
                else:
                    kw["timeout"] = 900
            elif framework == "autogluon":
                kw["time_limit"] = 900 if mode == "full" else 600
            return kw

        def _purge_bad_h2o():
            if not METRICS_JSON.exists():
                return
            rows = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
            kept = [r for r in rows if not (r.get("model_name") == "h2o_argmax" and is_degenerate_result(r))]
            if len(kept) != len(rows):
                METRICS_JSON.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

        def _already_done(model_name, mode_str, seed):
            if not METRICS_JSON.exists():
                return False
            for r in json.loads(METRICS_JSON.read_text(encoding="utf-8")):
                if (r.get("model_name"), r.get("mode"), r.get("seed")) != (model_name, mode_str, seed):
                    continue
                if (r.get("extra") or {{}}).get("prediction_mode") != "argmax":
                    continue
                if is_degenerate_result(r) or r.get("f1_oos") is None:
                    continue
                return True
            return False

        _purge_bad_h2o()

        if not PENDING_JOBS:
            print("RUN_FINISH " + json.dumps({{"status": "nothing_pending", "jobs": 0}}, ensure_ascii=False))
        else:
            total = len(PENDING_JOBS)
            for i, job in enumerate(PENDING_JOBS, 1):
                fw = job["framework"]
                mode = job["mode"]
                n_shots = job["n_shots"]
                seed = job["seed"]
                mode_str = job["mode_str"]
                model_name = MODEL_BY_FW[fw]
                meta = {{
                    "run": i, "total": total, "framework": fw, "mode": mode_str,
                    "n_shots": n_shots, "seed": seed, "prediction_mode": "argmax",
                    "embedder": EMBEDDER, "source": SOURCE,
                }}
                print("RUN_START " + json.dumps(meta, ensure_ascii=False))
                if _already_done(model_name, mode_str, seed):
                    print("RUN_FINISH " + json.dumps({{**meta, "status": "skipped"}}, ensure_ascii=False))
                    continue
                try:
                    run_single_experiment(
                        framework_name=fw,
                        source=SOURCE,
                        mode=mode,
                        n_shots=n_shots,
                        seed=seed,
                        results_file=METRICS_JSON,
                        wrapper_kwargs={{**_wrapper_kwargs(fw, mode, n_shots), "seed": seed}},
                        calibrate_threshold=False,
                        prediction_mode="argmax",
                    )
                except Exception as e:
                    print("RUN_FINISH " + json.dumps({{
                        **meta, "status": "failed",
                        "error_type": type(e).__name__, "error_message": str(e),
                    }}, ensure_ascii=False))
                    if fw == "h2o":
                        try:
                            import h2o
                            h2o.cluster().shutdown(prompt=False)
                        except Exception:
                            pass
        '''
    ).strip()

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            cell_md(md),
            cell_code(setup),
            cell_code(hf),
            cell_code(install_bundle),
            cell_code(prepare),
            cell_code(run),
        ],
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", OUT)
    print("cells:", len(nb["cells"]), "bundle_files:", len(bundle), "pending_jobs:", len(pending))
    for j in pending:
        print(f"  {j['framework']:10} {j['mode_str']:8} seed={j['seed']}")


if __name__ == "__main__":
    main()
