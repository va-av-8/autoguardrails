"""
Подготовка данных CLINC150 для экспериментов.

Загружает данные из HuggingFace, конвертирует в формат AutoIntent,
сохраняет в data/processed/.

Использование:
    python prepare_data.py --mode full
    python prepare_data.py --mode fewshot --n_shots 10 20 50

TODO:
    - [ ] Реализовать загрузку CLINC150 через datasets
    - [ ] Реализовать few-shot сэмплинг
    - [ ] Реализовать конвертацию в формат AutoIntent
    - [ ] Добавить валидацию данных
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare CLINC150 data")
    parser.add_argument("--mode", choices=["full", "fewshot"], default="full")
    parser.add_argument("--n_shots", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))

    args = parser.parse_args()

    # TODO: реализовать
    raise NotImplementedError("prepare_data.py not implemented yet")


if __name__ == "__main__":
    main()
