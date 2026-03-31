"""
Единый интерфейс оценки моделей.
Все модели (бейзлайн, SOTA, AutoIntent) оцениваются через один класс,
чтобы результаты были сопоставимы.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
from typing import Protocol, runtime_checkable

import numpy as np

from .metrics import compute_all_metrics, measure_latency


@runtime_checkable
class OOSModel(Protocol):
    """Protocol for OOS detection models."""

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return predicted labels (-1 for OOS, 0..N for intents)."""
        ...

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return OOS probability scores (higher = more likely OOS)."""
        ...


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
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


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
        """
        Args:
            test_data: {"texts": list[str], "labels": list[int]}
                       где -1 = OOS, 0..N = in-scope intents
            results_dir: директория для сохранения результатов
        """
        self.texts = test_data["texts"]
        self.labels = np.array(test_data["labels"])
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.results_dir / "metrics.json"
        self.oos_label = -1

    def evaluate(
        self,
        model: OOSModel,
        model_name: str,
        mode: str = "full",
        n_shots: int | None = None,
        seed: int | None = None,
        is_reference: bool = False,
        measure_latency_flag: bool = True,
    ) -> EvaluationResult:
        """
        Прогоняет модель на тест-сплите, вычисляет все метрики,
        измеряет latency.

        Args:
            model: объект с методами predict() и predict_proba_oos()
            model_name: название модели для отчёта
            mode: "full" | "10shot" | "20shot" | "50shot"
            n_shots: число примеров на класс (для few-shot)
            seed: random seed (для few-shot)
            is_reference: True для guardrail reference model
            measure_latency_flag: измерять ли latency

        Returns:
            EvaluationResult с заполненными метриками
        """
        # Get predictions
        y_pred = model.predict(self.texts)
        y_scores = model.predict_proba(self.texts)

        # Compute metrics
        metrics = compute_all_metrics(
            y_true=self.labels,
            y_scores=y_scores,
            y_pred=y_pred,
            oos_label=self.oos_label,
        )

        # Measure latency (first 100 texts for speed)
        latency = 0.0
        if measure_latency_flag:
            latency = measure_latency(model, self.texts[:100])

        return EvaluationResult(
            model_name=model_name,
            mode=mode,
            oos_recall=metrics["oos_recall"],
            in_domain_acc=metrics["in_domain_acc"],
            f1_oos=metrics["f1_oos"],
            auroc=metrics["auroc"],
            au_ioc=metrics["au_ioc"],
            latency_ms=latency,
            n_shots=n_shots,
            seed=seed,
            is_reference=is_reference,
        )

    def save(self, result: EvaluationResult) -> None:
        """Дописывает результат в results/metrics.json."""
        # Load existing results
        results = []
        if self.metrics_file.exists():
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                results = json.load(f)

        # Append new result
        results.append(result.to_dict())

        # Save
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def load_results(self) -> list[dict]:
        """Загружает все сохранённые результаты."""
        if not self.metrics_file.exists():
            return []
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def print_report(self) -> None:
        """Печатает сравнительную таблицу всех сохранённых результатов."""
        results = self.load_results()
        if not results:
            print("No results saved yet.")
            return

        # Header
        print("\n" + "=" * 100)
        print("OOS Detection Evaluation Report")
        print("=" * 100)
        print(
            f"{'Model':<25} {'Mode':<10} "
            f"{'OOS Rec':>8} {'InD Acc':>8} {'F1 OOS':>8} "
            f"{'AUROC':>8} {'AU-IOC':>8} {'Lat(ms)':>8}"
        )
        print("-" * 100)

        # Sort by f1_oos descending
        sorted_results = sorted(results, key=lambda x: -x["f1_oos"])

        for r in sorted_results:
            # Mark reference models
            name = r["model_name"]
            if r.get("is_reference"):
                name += " [ref]"

            print(
                f"{name:<25} {r['mode']:<10} "
                f"{r['oos_recall']:>8.4f} {r['in_domain_acc']:>8.4f} {r['f1_oos']:>8.4f} "
                f"{r['auroc']:>8.4f} {r['au_ioc']:>8.4f} {r['latency_ms']:>8.2f}"
            )

        print("-" * 100)
        print("* = Reference model (Chua et al. 2024)")
        print(
            "Primary metrics: OOS Recall, In-Domain Acc, F1 OOS | "
            "Secondary: AUROC, AU-IOC, Latency"
        )
        print("=" * 100 + "\n")
