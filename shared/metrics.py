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


def get_oos_scores_from_pipeline(pipeline, texts: list[str]) -> np.ndarray:
    """
    Get continuous OOS scores from AutoIntent pipeline.
    Uses predict_with_metadata to get class probability scores,
    then computes OOS score as 1 - max(class_scores).
    Falls back to binary scores if continuous unavailable.
    Returns array of shape (n_samples,).
    """
    if hasattr(pipeline, 'predict_with_metadata'):
        try:
            results = pipeline.predict_with_metadata(texts)
            if hasattr(results, 'utterances') and results.utterances:
                all_scores = [utt.score for utt in results.utterances
                              if hasattr(utt, 'score') and utt.score is not None]
                if all_scores:
                    return 1 - np.array(all_scores).max(axis=1)
        except Exception:
            pass
    # fallback
    preds = pipeline.predict(texts)
    return np.array([1.0 if p is None else 0.0 for p in preds])


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


# =============================================================================
# Jailbreak Detection Metrics
# =============================================================================


def over_refusal_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """
    Over-refusal Rate: доля safe-промптов, ошибочно заблокированных.

    FPR на safe (benign) промптах:
      = FP / (FP + TN)
    где positive = jailbreak/OOS,
    считается только на примерах с y_true != oos_label.

    Используется в:
    - WildGuard (NeurIPS 2024) как "Refusal Rate on benign prompts"
    - XSTest (NAACL 2024) как "Over-refusal Rate"

    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        oos_label: метка jailbreak/OOS класса

    Returns:
        over-refusal rate, float в [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Только safe примеры (y_true != oos_label)
    safe_mask = y_true != oos_label
    n_safe = safe_mask.sum()

    if n_safe == 0:
        return 0.0

    # FP: safe примеры, предсказанные как jailbreak
    false_positives = ((y_pred == oos_label) & safe_mask).sum()

    return float(false_positives / n_safe)


def precision_oos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """
    Precision на jailbreak/OOS-классе:
      TP / (TP + FP)
    Доля заблокированных промптов, которые действительно jailbreak.

    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        oos_label: метка jailbreak/OOS класса

    Returns:
        precision, float в [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Предсказанные как OOS
    pred_oos_mask = y_pred == oos_label
    n_pred_oos = pred_oos_mask.sum()

    if n_pred_oos == 0:
        return 0.0

    # TP: правильно предсказанные OOS
    true_positives = ((y_true == oos_label) & pred_oos_mask).sum()

    return float(true_positives / n_pred_oos)


def evaluate_jailbreak(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_types: np.ndarray | None = None,
    oos_label: int = 1,
) -> dict:
    """
    Полный набор метрик для Jailbreak Detection.

    Args:
        y_true: истинные метки (1 = jailbreak, 0 = safe)
        y_pred: предсказанные метки
        data_types: массив строк data_type из WildJailbreak.
                    Если None — метрики по подмножествам не считаются.
        oos_label: метка jailbreak-класса (по умолчанию 1)

    Returns dict:
        "f1"                         : f1_oos(...)
        "precision"                  : precision_oos(...)
        "recall"                     : oos_recall(...)
        "over_refusal_rate"          : over_refusal_rate(...)

        Если data_types передан, дополнительно:
        "recall_vanilla_harmful"     : recall на vanilla_harmful
        "recall_adversarial_harmful" : recall на adversarial_harmful
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {
        "f1": f1_oos(y_true, y_pred, oos_label),
        "precision": precision_oos(y_true, y_pred, oos_label),
        "recall": oos_recall(y_true, y_pred, oos_label),
        "over_refusal_rate": over_refusal_rate(y_true, y_pred, oos_label),
    }

    if data_types is not None:
        data_types = np.asarray(data_types)

        # Recall on vanilla_harmful subset
        vanilla_harmful_mask = data_types == "vanilla_harmful"
        if vanilla_harmful_mask.sum() > 0:
            result["recall_vanilla_harmful"] = oos_recall(
                y_true[vanilla_harmful_mask],
                y_pred[vanilla_harmful_mask],
                oos_label,
            )

        # Recall on adversarial_harmful subset
        adversarial_harmful_mask = data_types == "adversarial_harmful"
        if adversarial_harmful_mask.sum() > 0:
            result["recall_adversarial_harmful"] = oos_recall(
                y_true[adversarial_harmful_mask],
                y_pred[adversarial_harmful_mask],
                oos_label,
            )

    return result


if __name__ == "__main__":
    # Smoke test for over_refusal_rate
    y_true = np.array([0, 0, 0, 0, 1, 1])  # 4 safe, 2 jailbreak
    y_pred = np.array([0, 1, 0, 1, 1, 1])  # 2 safe ошибочно заблокированы
    orr = over_refusal_rate(y_true, y_pred, oos_label=1)
    assert orr == 0.5, f"Expected 0.5, got {orr}"
    print(f"over_refusal_rate test passed: {orr}")

    # Smoke test for precision_oos
    # y_pred has 4 predictions as jailbreak (indices 1, 3, 4, 5)
    # y_true jailbreak at indices 4, 5 → TP=2, FP=2 → precision=0.5
    prec = precision_oos(y_true, y_pred, oos_label=1)
    assert prec == 0.5, f"Expected 0.5, got {prec}"
    print(f"precision_oos test passed: {prec}")

    # Smoke test for evaluate_jailbreak
    metrics = evaluate_jailbreak(y_true, y_pred, oos_label=1)
    print(f"evaluate_jailbreak test passed: {metrics}")
