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
    # TODO: реализовать
    raise NotImplementedError


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
    # TODO: реализовать
    raise NotImplementedError


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
    # TODO: реализовать
    raise NotImplementedError


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
    # TODO: реализовать
    raise NotImplementedError


def au_ioc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
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
        oos_label: метка OOS-класса
        n_thresholds: число точек для построения кривой

    Returns:
        AU-IOC, float в [0, 1]
    """
    # TODO: реализовать
    raise NotImplementedError


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
    # TODO: реализовать
    raise NotImplementedError


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
    # TODO: реализовать
    raise NotImplementedError
