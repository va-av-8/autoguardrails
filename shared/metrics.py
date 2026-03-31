"""
Метрики для оценки OOS-детекции.

Основные (прямое сравнение с литературой):
  - oos_recall     : TPR на OOS-классе (ADB, DETER, AutoIntent)
  - in_domain_acc  : accuracy на in-scope (AutoIntent Table 3: 96.13)
  - f1_oos         : F1 на OOS-классе (AutoIntent Table 3: 76.79)

Вспомогательные:
  - auroc          : Area Under ROC Curve (EMNLP Industry 2024)
  - au_ioc         : Area Under IOC curve (Springer Applied Intelligence 2024)
  - latency_ms     : время инференса (для guardrail-аргументации)
"""

from __future__ import annotations
import time

import numpy as np
from sklearn.metrics import (
    auc,
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
)


def oos_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """Recall на OOS-классе (TPR)."""
    y_true_bin = (np.asarray(y_true) == oos_label).astype(int)
    y_pred_bin = (np.asarray(y_pred) == oos_label).astype(int)
    return float(recall_score(y_true_bin, y_pred_bin, zero_division=0))


def in_domain_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """Accuracy только на in-scope примерах."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != oos_label
    if mask.sum() == 0:
        return 0.0
    return float(accuracy_score(y_true[mask], y_pred[mask]))


def f1_oos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """F1-score на OOS-классе."""
    y_true_bin = (np.asarray(y_true) == oos_label).astype(int)
    y_pred_bin = (np.asarray(y_pred) == oos_label).astype(int)
    return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))


def auroc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    oos_label: int = -1,
) -> float:
    """Area Under ROC Curve для бинарной задачи OOS vs in-scope."""
    y_true_binary = (np.asarray(y_true) == oos_label).astype(int)
    if len(np.unique(y_true_binary)) < 2:
        return 0.5
    return float(roc_auc_score(y_true_binary, y_scores))


def au_ioc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred_intent: np.ndarray | None = None,
    oos_label: int = -1,
    n_thresholds: int = 100,
) -> float:
    """
    Area Under In-scope/Out-of-scope Characteristic curve.
    X-axis: in-domain accuracy, Y-axis: OOS recall.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    oos_mask = y_true == oos_label
    inscope_mask = ~oos_mask
    n_oos, n_inscope = oos_mask.sum(), inscope_mask.sum()

    if n_oos == 0 or n_inscope == 0:
        return 0.5

    # Vectorized: thresholds x samples
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_thresholds)
    pred_oos = y_scores[None, :] >= thresholds[:, None]  # (n_thresh, n_samples)

    # OOS recall per threshold
    oos_recalls = (pred_oos & oos_mask).sum(axis=1) / n_oos

    # In-domain accuracy per threshold
    inscope_not_oos = inscope_mask & ~pred_oos  # (n_thresh, n_samples)
    if y_pred_intent is not None:
        y_pred_intent = np.asarray(y_pred_intent)
        correct = inscope_not_oos & (y_pred_intent == y_true)
        in_domain_accs = correct.sum(axis=1) / n_inscope
    else:
        in_domain_accs = inscope_not_oos.sum(axis=1) / n_inscope

    # Sort by x and compute AUC
    order = np.argsort(in_domain_accs)
    auc_value = float(auc(in_domain_accs[order], oos_recalls[order]))
    return max(0.0, min(1.0, auc_value))


def measure_latency(
    model,
    texts: list[str],
    n_warmup: int = 5,
    n_runs: int = 50,
) -> float:
    """Среднее время инференса на 1 запрос в миллисекундах."""
    texts = texts[:100]

    # Warmup
    for i in range(n_warmup):
        model.predict([texts[i % len(texts)]])

    # Measure
    times = []
    for i in range(n_runs):
        text = texts[i % len(texts)]
        start = time.perf_counter()
        model.predict([text])
        times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times))


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> dict:
    """Вычисляет все метрики и возвращает словарь."""
    return {
        "oos_recall": oos_recall(y_true, y_pred, oos_label),
        "in_domain_acc": in_domain_accuracy(y_true, y_pred, oos_label),
        "f1_oos": f1_oos(y_true, y_pred, oos_label),
        "auroc": auroc(y_true, y_scores, oos_label),
        "au_ioc": au_ioc(y_true, y_scores, y_pred, oos_label),
    }
