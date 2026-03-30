"""Визуализация результатов для guardrail-экспериментов."""

from __future__ import annotations
import numpy as np


def plot_roc_curves(results: list[dict], save_path=None) -> None:
    """
    ROC-кривые для всех моделей на одном графике.
    Отмечает рабочие точки FPR=0.05 и FPR=0.10.
    """
    # TODO: реализовать
    raise NotImplementedError


def plot_fewshot_scaling(results: list[dict], save_path=None) -> None:
    """
    Зависимость Recall@FPR от количества few-shot примеров.
    Основной график для демонстрации production-применимости AutoIntent.
    X-ось: n_shots (10, 20, 50, full), Y-ось: Recall@FPR=0.10
    """
    # TODO: реализовать
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, save_path=None) -> None:
    """Матрица ошибок с выделением OOS-строки и столбца."""
    # TODO: реализовать
    raise NotImplementedError


def plot_threshold_curve(y_true, y_scores, save_path=None) -> None:
    """
    Precision/Recall/F1 как функция порога.
    Помогает выбрать рабочую точку под конкретный сценарий.
    """
    # TODO: реализовать
    raise NotImplementedError
