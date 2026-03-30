"""
Метрики для оценки guardrail-моделей.

В отличие от стандартных метрик классификации, здесь приоритет на
recall при фиксированном FPR — это соответствует production-требованиям
guardrail: пропустить нарушение дороже, чем ложно заблокировать запрос.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score


def recall_at_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fpr_target: float = 0.10,
    oos_label: int = -1,
) -> float:
    """
    Recall на OOS-классе при фиксированном FPR на in-scope классах.

    Это основная метрика для guardrail-сценария:
    - FPR = доля легитимных запросов, ложно заблокированных
    - Recall = доля OOS-запросов, корректно пойманных

    Args:
        y_true: истинные метки (-1 для OOS, 0..N для in-scope)
        y_scores: скоры вероятности OOS-класса (выше = более OOS)
        fpr_target: целевой FPR (0.05 или 0.10)
        oos_label: метка OOS-класса в y_true

    Returns:
        recall на OOS при заданном FPR
    """
    # TODO: реализовать
    raise NotImplementedError


def auroc_oos(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    oos_label: int = -1,
) -> float:
    """
    AUROC для бинарной задачи OOS vs in-scope.

    Args:
        y_true: истинные метки
        y_scores: скоры OOS-класса
        oos_label: метка OOS в y_true

    Returns:
        значение AUROC
    """
    # TODO: реализовать
    raise NotImplementedError


def f1_oos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oos_label: int = -1,
) -> float:
    """
    F1 на OOS-классе. Используется для сравнения с результатами
    из статьи AutoIntent (Table 3: OOS F1 = 76.79 на CLINC150).

    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        oos_label: метка OOS в y_true

    Returns:
        F1-score для OOS-класса
    """
    # TODO: реализовать
    raise NotImplementedError


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    fpr_targets: list[float] | None = None,
    oos_label: int = -1,
) -> dict:
    """
    Вычисляет все метрики разом и возвращает словарь.
    Используется в evaluate.py для единообразного репортинга.

    Returns:
        dict с ключами: recall_at_fpr_05, recall_at_fpr_10,
                        auroc, f1_oos
    """
    # TODO: реализовать
    raise NotImplementedError
