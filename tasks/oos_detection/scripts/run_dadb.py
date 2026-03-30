"""
Запуск DA-ADB (Distance-Aware Adaptive Decision Boundary).

SOTA метод для OOS-детекции (Zhang et al., 2021).
Используется как верхняя граница качества.

Использование:
    python run_dadb.py --config configs/dadb.yaml

TODO:
    - [ ] Интеграция с репозиторием DA-ADB
    - [ ] Обучение модели
    - [ ] Оценка на тест-сплите
    - [ ] Сохранение результатов
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run DA-ADB")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))

    args = parser.parse_args()

    # TODO: реализовать
    raise NotImplementedError("run_dadb.py not implemented yet")


if __name__ == "__main__":
    main()
