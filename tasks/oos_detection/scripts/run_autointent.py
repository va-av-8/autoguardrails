"""
Запуск AutoIntent для OOS-детекции.

Режимы:
- fewshot: n примеров на intent (основной эксперимент)
- full: полный train-сплит (воспроизведение результатов статьи)

Использование:
    python run_autointent.py --mode fewshot --n_shots 10 20 50
    python run_autointent.py --mode full
    python run_autointent.py --mode fewshot --hypothesis per_intent_threshold

TODO:
    - [ ] Загрузка данных в формате AutoIntent
    - [ ] Конфигурация AutoIntent pipeline
    - [ ] Запуск оптимизации
    - [ ] Оценка на тест-сплите
    - [ ] Реализация HYP-001 (per-intent threshold)
    - [ ] Сохранение результатов
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run AutoIntent")
    parser.add_argument("--mode", choices=["fewshot", "full"], required=True)
    parser.add_argument("--n_shots", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--hypothesis", type=str, default=None,
                        help="Hypothesis to test (e.g., per_intent_threshold)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))

    args = parser.parse_args()

    # TODO: реализовать
    raise NotImplementedError("run_autointent.py not implemented yet")


if __name__ == "__main__":
    main()
