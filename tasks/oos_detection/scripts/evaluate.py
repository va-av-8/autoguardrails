"""
Единый скрипт оценки моделей.

Загружает предсказания модели и вычисляет все метрики:
- Recall@FPR=0.05, Recall@FPR=0.10
- AUROC
- F1 (OOS)
- Latency

Использование:
    python evaluate.py --predictions results/predictions.json
    python evaluate.py --predictions results/predictions.json --save_errors

TODO:
    - [ ] Загрузка предсказаний
    - [ ] Вычисление всех метрик
    - [ ] Генерация error_analysis.csv
    - [ ] Обновление metrics.json
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test_data", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--save_errors", action="store_true",
                        help="Save error analysis to CSV")

    args = parser.parse_args()

    # TODO: реализовать
    raise NotImplementedError("evaluate.py not implemented yet")


if __name__ == "__main__":
    main()
