"""
Единый интерфейс оценки моделей.
Все модели (бейзлайн, SOTA, AutoIntent) оцениваются через один класс,
чтобы результаты были сопоставимы.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import time


@dataclass
class EvaluationResult:
    """
    Результат оценки одной модели на тест-сплите.

    Метрики соответствуют стандарту OOS NLP-литературы.
    Основные три (oos_recall, in_domain_acc, f1_oos) позволяют
    напрямую сравниться с AutoIntent Table 3, ADB, DETER.
    """
    model_name: str
    mode: str                      # "full" | "10shot" | "20shot" | "50shot"

    # Основные метрики — прямое сравнение с литературой
    oos_recall: float = 0.0        # TPR на OOS; стандарт в ADB, DETER, AutoIntent
    in_domain_acc: float = 0.0     # accuracy на in-scope; AutoIntent Table 3: 96.13
    f1_oos: float = 0.0            # F1 на OOS; AutoIntent Table 3: 76.79

    # Вспомогательные метрики
    auroc: float = 0.0             # EMNLP Industry 2024
    au_ioc: float = 0.0            # Springer Applied Intelligence 2024
    latency_ms: float = 0.0        # для guardrail-аргументации

    # Мета-информация
    n_shots: int | None = None     # None = full train
    seed: int | None = None
    is_reference: bool = False     # True для guardrail reference (Chua 2024)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # TODO: реализовать
        raise NotImplementedError


class Evaluator:
    """
    Оценивает модель на тест-сплите и сохраняет результаты.

    Использование:
        evaluator = Evaluator(test_data, results_dir)
        result = evaluator.evaluate(model, model_name="autointent_10shot")
        evaluator.save(result)
        evaluator.print_report()
    """

    def __init__(self, test_data: dict, results_dir: Path):
        # TODO: реализовать
        raise NotImplementedError

    def evaluate(self, model, model_name: str, mode: str = "full",
                 n_shots: int | None = None) -> EvaluationResult:
        """
        Прогоняет модель на тест-сплите, вычисляет все метрики,
        измеряет latency.
        """
        # TODO: реализовать
        raise NotImplementedError

    def save(self, result: EvaluationResult) -> None:
        """Дописывает результат в results/metrics.json."""
        # TODO: реализовать
        raise NotImplementedError

    def print_report(self) -> None:
        """Печатает сравнительную таблицу всех сохранённых результатов."""
        # TODO: реализовать
        raise NotImplementedError
