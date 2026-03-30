"""
Утилиты для загрузки и подготовки данных.

Поддерживает два режима:
- full: стандартный train/val/test сплит CLINC150
- fewshot: сэмплинг n примеров на intent из train,
           тест-сплит всегда фиксированный
"""

from __future__ import annotations
from pathlib import Path
import numpy as np


def load_clinc150(split: str = "test", oos_difficulty: str = "plus") -> dict:
    """
    Загружает CLINC150 через HuggingFace datasets.

    Args:
        split: "train" | "validation" | "test"
        oos_difficulty: "small" | "imbalanced" | "plus" (plus = все три уровня OOS)

    Returns:
        dict с ключами: texts (list[str]), labels (list[int]),
                        oos_mask (list[bool])
    """
    # TODO: реализовать через datasets.load_dataset("clinc_oos", oos_difficulty)
    raise NotImplementedError


def sample_fewshot(
    data: dict,
    n_shots: int,
    oos_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Сэмплирует n примеров на intent из обучающей выборки.
    OOS-примеры добавляются пропорционально: n_intents * n_shots * oos_ratio.

    Args:
        data: выход load_clinc150
        n_shots: количество примеров на intent
        oos_ratio: доля OOS от общего числа примеров
        seed: random seed для воспроизводимости

    Returns:
        dict того же формата, что и data
    """
    # TODO: реализовать
    raise NotImplementedError


def convert_to_autointent_format(data: dict) -> dict:
    """
    Конвертирует данные в формат AutoIntent Dataset.

    Согласно документации AutoIntent, OOS-примеры передаются
    без поля 'label' (просто {"utterance": "..."}),
    in-scope примеры — с полем 'label'.

    Args:
        data: выход load_clinc150 или sample_fewshot

    Returns:
        dict совместимый с autointent.Dataset.from_dict()
    """
    # TODO: реализовать
    raise NotImplementedError


def save_processed(data: dict, path: Path) -> None:
    """Сохраняет обработанные данные в jsonl."""
    # TODO: реализовать
    raise NotImplementedError


def load_processed(path: Path) -> dict:
    """Загружает обработанные данные из jsonl."""
    # TODO: реализовать
    raise NotImplementedError
