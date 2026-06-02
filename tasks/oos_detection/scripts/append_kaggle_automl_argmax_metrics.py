#!/usr/bin/env python3
"""Append successful Kaggle argmax AutoML runs to results/metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
METRICS_FILE = TASK_DIR / "results" / "metrics.json"

# RUN_FINISH payloads with status ok (from Kaggle log 2026-05-25)
KAGGLE_OK = [
    {"model_name": "autogluon_argmax", "mode": "full", "oos_recall": 0.573, "in_domain_acc": 0.9633333333333334, "f1_oos": 0.7266962587190868, "auroc": 0.9647573333333332, "au_ioc": 0.883350888888889, "latency_ms": 152.5923674599835, "n_shots": None, "seed": 42, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 428.7, "kaggle_run": 1}},
    {"model_name": "autogluon_argmax", "mode": "full", "oos_recall": 0.55, "in_domain_acc": 0.9644444444444444, "f1_oos": 0.7051282051282052, "auroc": 0.9525959999999999, "au_ioc": 0.8881912222222221, "latency_ms": 156.5355473999989, "n_shots": None, "seed": 123, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 428.5, "kaggle_run": 2}},
    {"model_name": "autogluon_argmax", "mode": "full", "oos_recall": 0.54, "in_domain_acc": 0.9542222222222222, "f1_oos": 0.697224015493867, "auroc": 0.9706048888888887, "au_ioc": 0.8269162222222223, "latency_ms": 44.70735128001252, "n_shots": None, "seed": 456, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 419.1, "kaggle_run": 3}},
    {"model_name": "autogluon_argmax", "mode": "10shot", "oos_recall": 0.633, "in_domain_acc": 0.9077777777777778, "f1_oos": 0.7447058823529412, "auroc": 0.9423755555555555, "au_ioc": 0.8098288888888888, "latency_ms": 40.99401830003444, "n_shots": 10, "seed": 42, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 414.1, "kaggle_run": 4}},
    {"model_name": "autogluon_argmax", "mode": "10shot", "oos_recall": 0.736, "in_domain_acc": 0.8924444444444445, "f1_oos": 0.7935309973045822, "auroc": 0.9280164444444445, "au_ioc": 0.8394044444444444, "latency_ms": 283.96999667989803, "n_shots": 10, "seed": 123, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 411.4, "kaggle_run": 5}},
    {"model_name": "autogluon_argmax", "mode": "10shot", "oos_recall": 0.644, "in_domain_acc": 0.9257777777777778, "f1_oos": 0.7532163742690059, "auroc": 0.9458755555555556, "au_ioc": 0.8276373333333333, "latency_ms": 40.015841479980736, "n_shots": 10, "seed": 456, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 415.5, "kaggle_run": 6}},
    {"model_name": "autogluon_argmax", "mode": "20shot", "oos_recall": 0.617, "in_domain_acc": 0.9355555555555556, "f1_oos": 0.7420324714371618, "auroc": 0.9585937777777779, "au_ioc": 0.8405437777777778, "latency_ms": 40.63041831997907, "n_shots": 20, "seed": 42, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 415.1, "kaggle_run": 7}},
    {"model_name": "autogluon_argmax", "mode": "20shot", "oos_recall": 0.747, "in_domain_acc": 0.9264444444444444, "f1_oos": 0.83, "auroc": 0.949597111111111, "au_ioc": 0.885561111111111, "latency_ms": 231.31116888003817, "n_shots": 20, "seed": 123, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 414.7, "kaggle_run": 8}},
    {"model_name": "autogluon_argmax", "mode": "20shot", "oos_recall": 0.593, "in_domain_acc": 0.9288888888888889, "f1_oos": 0.7280540208717005, "auroc": 0.9390508888888889, "au_ioc": 0.8102424444444445, "latency_ms": 42.47405724003329, "n_shots": 20, "seed": 456, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 418.2, "kaggle_run": 9}},
    {"model_name": "autogluon_argmax", "mode": "50shot", "oos_recall": 0.654, "in_domain_acc": 0.9548888888888889, "f1_oos": 0.7827648114901257, "auroc": 0.9599535555555556, "au_ioc": 0.8986056666666667, "latency_ms": 192.0494741800394, "n_shots": 50, "seed": 42, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 415.9, "kaggle_run": 10}},
    {"model_name": "autogluon_argmax", "mode": "50shot", "oos_recall": 0.58, "in_domain_acc": 0.9477777777777778, "f1_oos": 0.7268170426065163, "auroc": 0.9528511111111111, "au_ioc": 0.893224, "latency_ms": 183.55183199999374, "n_shots": 50, "seed": 123, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 415.2, "kaggle_run": 11}},
    {"model_name": "autogluon_argmax", "mode": "50shot", "oos_recall": 0.636, "in_domain_acc": 0.9553333333333334, "f1_oos": 0.7709090909090909, "auroc": 0.964934888888889, "au_ioc": 0.8737872222222222, "latency_ms": 43.062639360014145, "n_shots": 50, "seed": 456, "is_reference": False, "extra": {"framework": "autogluon", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 418.0, "kaggle_run": 12}},
    {"model_name": "lama_argmax", "mode": "full", "oos_recall": 0.618, "in_domain_acc": 0.9713333333333334, "f1_oos": 0.759680393362016, "auroc": 0.9782144444444444, "au_ioc": 0.9121432222222222, "latency_ms": 512.519293360092, "n_shots": None, "seed": 42, "is_reference": False, "extra": {"framework": "lama", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 1612.8, "kaggle_run": 25}},
    {"model_name": "lama_argmax", "mode": "full", "oos_recall": 0.615, "in_domain_acc": 0.972, "f1_oos": 0.7564575645756457, "auroc": 0.9766753333333333, "au_ioc": 0.8871871111111111, "latency_ms": 513.6227138400136, "n_shots": None, "seed": 456, "is_reference": False, "extra": {"framework": "lama", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 1608.0, "kaggle_run": 27}},
    {"model_name": "lama_argmax", "mode": "10shot", "oos_recall": 0.681, "in_domain_acc": 0.9142222222222223, "f1_oos": 0.7747440273037542, "auroc": 0.9466146666666665, "au_ioc": 0.8449156666666666, "latency_ms": 501.1768719400425, "n_shots": 10, "seed": 42, "is_reference": False, "extra": {"framework": "lama", "source": "clinc_oos_plus", "prediction_mode": "argmax", "embedder": "intfloat/multilingual-e5-large-instruct", "fit_sec": 4346.3, "kaggle_run": 28}},
]


def _key(r: dict) -> tuple:
    extra = r.get("extra") or {}
    return (r["model_name"], r["mode"], r.get("seed"), extra.get("source"))


def main() -> None:
    results = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    existing = {_key(r) for r in results}
    added = 0
    for rec in KAGGLE_OK:
        if _key(rec) in existing:
            continue
        results.append(rec)
        existing.add(_key(rec))
        added += 1
    METRICS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Appended {added} records (total {len(results)})")


if __name__ == "__main__":
    main()
