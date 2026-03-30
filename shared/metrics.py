"""
Метрики для оценки OOS-детекции.

Схема метрик соответствует академической OOS-литературе:

Основные (прямое сравнение с опубликованными результатами):
  - oos_recall     : TPR на OOS-классе
                     Стандарт в ADB (AAAI 2021), DETER (2024),
                     AutoIntent (EMNLP 2025) и большинстве OOS-работ.
                     Мотивация: пропустить OOS-запрос (FN) хуже,
                     чем ложно заблокировать in-scope (FP) —
                     FP вызывает fallback, FN даёт неверный ответ.

  - in_domain_acc  : accuracy только на in-scope примерах.
                     Название совпадает с AutoIntent Table 3 (96.13).
                     Синоним: in-scope accuracy — одна и та же метрика.

  - f1_oos         : F1 на OOS-классе.
                     Прямое сравнение с AutoIntent Table 3 (76.79)
                     и результатами ADB, DETER на CLINC150.

Вспомогательные:
  - auroc          : Area Under ROC Curve.
                     Используется в "Intent Detection in the Age of LLMs"
                     (EMNLP Industry 2024) как дополнительная метрика.

  - au_ioc         : Area Under In-scope/Out-of-scope Characteristic curve.
                     Предложена в Springer Applied Intelligence (2024)
                     специально для OOS-задачи.
                     Кривая: X = in-domain accuracy, Y = OOS recall
                     при изменении порога. Отражает guardrail trade-off:
                     баланс между защитой (OOS recall) и usability
                     (не блокировать легитимные запросы).

  - latency_ms     : время инференса на 1 запрос в миллисекундах.
                     Нет прецедента в OOS NLP-литературе, но необходима
                     для guardrail-аргументации (guardrail в критическом
                     пути каждого запроса).

Намеренно исключено:
  - recall_at_fpr  : нет прецедента в академической OOS NLP-литературе.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)


def oos_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """
    Recall на OOS-классе (TPR): доля OOS-запросов, корректно пойманных.

    Основная метрика детекции в OOS NLP-литературе.
    Высокий recall = мало пропущенных нарушений мандата.

    Args:
        y_true: истинные метки (-1 для OOS, 0..N для in-scope)
        y_pred: предсказанные метки
        oos_label: метка OOS-класса в y_true (по умолчанию -1)

    Returns:
        recall на OOS-классе, float в [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    oos_mask = y_true == oos_label
    if oos_mask.sum() == 0:
        return 0.0

    # TP: correctly predicted as OOS
    tp = ((y_true == oos_label) & (y_pred == oos_label)).sum()
    # FN: OOS but predicted as in-scope
    fn = ((y_true == oos_label) & (y_pred != oos_label)).sum()

    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def in_domain_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """
    Accuracy только на in-scope примерах (исключая OOS).

    Название совпадает с AutoIntent Table 3 — позволяет
    напрямую сравнить с опубликованным результатом (96.13).
    Синоним в литературе: in-scope accuracy.

    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        oos_label: метка OOS-класса (исключается из расчёта)

    Returns:
        accuracy на in-scope подмножестве, float в [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Filter to in-scope examples only
    inscope_mask = y_true != oos_label
    if inscope_mask.sum() == 0:
        return 0.0

    y_true_inscope = y_true[inscope_mask]
    y_pred_inscope = y_pred[inscope_mask]

    return float(accuracy_score(y_true_inscope, y_pred_inscope))


def f1_oos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """
    F1-score на OOS-классе.

    Прямое сравнение с:
    - AutoIntent Table 3: OOS F1 = 76.79 на CLINC150
    - ADB (AAAI 2021): F1 на open-class
    - DETER (2024): F1 на unknown-class

    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        oos_label: метка OOS-класса

    Returns:
        F1-score для OOS-класса, float в [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Convert to binary: OOS=1, in-scope=0
    y_true_binary = (y_true == oos_label).astype(int)
    y_pred_binary = (y_pred == oos_label).astype(int)

    # F1 for the positive class (OOS)
    return float(f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0))


def auroc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    oos_label: int = -1,
) -> float:
    """
    Area Under ROC Curve для бинарной задачи OOS vs in-scope.

    Используется в "Intent Detection in the Age of LLMs" (EMNLP 2024)
    как вспомогательная метрика наряду с F1 и OOS recall.
    Не зависит от порога — оценивает качество ранжирования.

    Args:
        y_true: истинные метки
        y_scores: скоры вероятности OOS (выше = более OOS)
        oos_label: метка OOS-класса

    Returns:
        AUROC, float в [0.5, 1.0]
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Convert to binary: OOS=1, in-scope=0
    y_true_binary = (y_true == oos_label).astype(int)

    # Check for degenerate cases
    if len(np.unique(y_true_binary)) < 2:
        return 0.5  # Random baseline

    return float(roc_auc_score(y_true_binary, y_scores))


def au_ioc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred_intent: np.ndarray | None = None,
    oos_label: int = -1,
    n_thresholds: int = 100,
) -> float:
    """
    Area Under In-scope/Out-of-scope Characteristic curve (AU-IOC).

    Предложена в: Few-shot out-of-scope intent classification:
    analyzing the robustness of prompt-based learning.
    Springer Applied Intelligence, 2024.

    Кривая строится по изменению порога:
      X-ось: in-domain accuracy (доля корректно классифицированных in-scope)
      Y-ось: OOS recall (доля пойманных OOS)

    AU-IOC отражает guardrail trade-off: насколько хорошо модель
    одновременно ловит OOS и не блокирует легитимные запросы.
    Чем выше — тем лучше баланс защиты и usability.

    В отличие от AUROC (OOS vs in-scope бинарно), AU-IOC учитывает
    качество классификации внутри in-scope классов.

    Args:
        y_true: истинные метки (-1 для OOS, 0..N для in-scope intent)
        y_scores: скоры OOS-вероятности (выше = более OOS)
        y_pred_intent: предсказанные intent-метки (если None, используется
                       упрощённая версия - просто accuracy на не-OOS)
        oos_label: метка OOS-класса
        n_thresholds: число точек для построения кривой

    Returns:
        AU-IOC, float в [0, 1]
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if y_pred_intent is not None:
        y_pred_intent = np.asarray(y_pred_intent)

    # Generate thresholds from score distribution
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_thresholds)

    # Masks
    oos_mask = y_true == oos_label
    inscope_mask = ~oos_mask

    n_oos = oos_mask.sum()
    n_inscope = inscope_mask.sum()

    if n_oos == 0 or n_inscope == 0:
        return 0.5  # Degenerate case

    x_points = []  # in-domain accuracy
    y_points = []  # OOS recall

    for thresh in thresholds:
        # Predict OOS if score >= threshold
        pred_oos = y_scores >= thresh

        # OOS recall: TP / (TP + FN)
        oos_tp = (oos_mask & pred_oos).sum()
        oos_recall_val = oos_tp / n_oos

        # In-domain accuracy: correct in-scope predictions among in-scope samples
        # An in-scope sample is "correct" if:
        # 1. It's NOT predicted as OOS (threshold)
        # 2. AND (if y_pred_intent provided) its intent is correctly predicted
        inscope_not_oos = inscope_mask & ~pred_oos

        if y_pred_intent is not None:
            # Check intent correctness for in-scope samples not flagged as OOS
            inscope_correct = (
                inscope_not_oos & (y_pred_intent == y_true)
            ).sum()
        else:
            # Simplified: just count in-scope not flagged as OOS
            inscope_correct = inscope_not_oos.sum()

        in_domain_acc_val = inscope_correct / n_inscope

        x_points.append(in_domain_acc_val)
        y_points.append(oos_recall_val)

    # Sort by x for proper AUC calculation
    sorted_indices = np.argsort(x_points)
    x_sorted = np.array(x_points)[sorted_indices]
    y_sorted = np.array(y_points)[sorted_indices]

    # Calculate AUC using trapezoidal rule
    auc = float(np.trapz(y_sorted, x_sorted))

    # Normalize to [0, 1] (max possible area is 1x1)
    return max(0.0, min(1.0, auc))


def measure_latency(
    model,
    texts: list[str],
    n_warmup: int = 5,
    n_runs: int = 100,
) -> float:
    """
    Среднее время инференса на 1 запрос в миллисекундах.

    Нет прецедента в OOS NLP-литературе, но важна для
    guardrail-аргументации: guardrail стоит в критическом
    пути каждого запроса к агентной системе.

    Args:
        model: объект с методом predict(texts)
        texts: список тестовых текстов
        n_warmup: прогревочные запуски (не считаются)
        n_runs: число измерений для усреднения

    Returns:
        среднее время на 1 запрос в миллисекундах
    """
    import time

    # Warmup runs
    for _ in range(n_warmup):
        _ = model.predict(texts[:1])

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(texts[:1])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return float(np.mean(times))


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> dict:
    """
    Вычисляет все метрики и возвращает словарь.
    Используется в Evaluator для единообразного репортинга.

    Args:
        y_true: истинные метки
        y_scores: OOS-скоры (для auroc, au_ioc)
        y_pred: предсказанные метки (для recall, accuracy, f1)
        oos_label: метка OOS-класса

    Returns:
        dict с ключами:
          oos_recall, in_domain_acc, f1_oos,  <- основные
          auroc, au_ioc                        <- вспомогательные
        latency_ms считается отдельно через measure_latency()
    """
    return {
        # Primary metrics
        "oos_recall": oos_recall(y_true, y_pred, oos_label),
        "in_domain_acc": in_domain_accuracy(y_true, y_pred, oos_label),
        "f1_oos": f1_oos(y_true, y_pred, oos_label),
        # Secondary metrics
        "auroc": auroc(y_true, y_scores, oos_label),
        "au_ioc": au_ioc(y_true, y_scores, y_pred, oos_label),
    }
