"""
Few-shot сэмплинг для экспериментов.

Сэмплирует n примеров на intent из train-сплита CLINC150.
OOS-примеры добавляются пропорционально.

Использование:
    python fewshot_sampling.py --n_shots 10 --seed 42
    python fewshot_sampling.py --n_shots 10 20 50 --seeds 42 123 456

TODO:
    - [ ] Стратифицированный сэмплинг по интентам
    - [ ] Контроль OOS-ratio
    - [ ] Сохранение в формате AutoIntent
    - [ ] Поддержка множественных seeds для усреднения
"""

from __future__ import annotations

import argparse
from pathlib import Path


def sample_fewshot(
    data: dict,
    n_shots: int,
    oos_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Сэмплирует n примеров на intent.

    Args:
        data: полный train-сплит
        n_shots: количество примеров на intent
        oos_ratio: доля OOS примеров
        seed: random seed

    Returns:
        few-shot подмножество данных
    """
    # TODO: реализовать
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Few-shot sampling")
    parser.add_argument("--n_shots", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--oos_ratio", type=float, default=0.1)
    parser.add_argument("--input_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/fewshot"))

    args = parser.parse_args()

    # TODO: реализовать
    raise NotImplementedError("fewshot_sampling.py not implemented yet")


if __name__ == "__main__":
    main()
