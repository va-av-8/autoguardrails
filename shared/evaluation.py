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
    model_name: str
    mode: str                    # "full" | "10shot" | "20shot" | "50shot"
    recall_at_fpr_05: float = 0.0
    recall_at_fpr_10: float = 0.0
    auroc: float = 0.0
    f1_oos: float = 0.0
    latency_ms: float = 0.0
    n_shots: int | None = None
    seed: int | None = None
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
