"""
Запуск бейзлайн-моделей для OOS-детекции.

Бейзлайны:
- TF-IDF + Logistic Regression (OOS как доп. класс)
- Cosine Similarity Threshold (embedding-based)

Использование:
    python run_baseline.py --model tfidf --config configs/baseline_tfidf.yaml
    python run_baseline.py --model cosine --config configs/baseline_cosine.yaml

TODO:
    - [ ] Загрузка данных
    - [ ] Обучение TF-IDF + LogReg
    - [ ] Реализация cosine threshold
    - [ ] Оценка на тест-сплите
    - [ ] Сохранение результатов
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--model", choices=["tfidf", "cosine"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))

    args = parser.parse_args()

    # TODO: реализовать
    raise NotImplementedError("run_baseline.py not implemented yet")


if __name__ == "__main__":
    main()
