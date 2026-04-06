"""
Утилиты для загрузки подготовленных данных CLINC150.

Данные должны быть предварительно созданы через:
    python scripts/prepare_data.py

Файлы в data/processed/:
    clinc150_full.json     — полные сплиты train/validation/test
    clinc150_fewshot.json  — few-shot выборки для разных n и seed
    clinc150_meta.json     — метаданные датасета
"""

from __future__ import annotations

import json
from pathlib import Path


def load_clinc150(split: str, processed_dir: Path) -> dict:
    """
    Загружает сплит CLINC150 в standard формате.

    Args:
        split: "train" | "validation" | "test"
        processed_dir: путь к data/processed/

    Returns:
        {"texts": list[str], "labels": list[int]}
        OOS имеет label = -1
    """
    full_path = processed_dir / "clinc150_full.json"
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["standard"][split]


def load_clinc150_autointent(split: str, processed_dir: Path) -> list[dict]:
    """
    Загружает сплит CLINC150 в формате AutoIntent.

    Args:
        split: "train" | "validation" | "test"
        processed_dir: путь к data/processed/

    Returns:
        list[dict] — in-scope: {"utterance": str, "label": int}
                     OOS: {"utterance": str} (без поля label)
    """
    full_path = processed_dir / "clinc150_full.json"
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["autointent"][split]


def load_fewshot(n_shots: int, seed: int, processed_dir: Path) -> dict:
    """
    Загружает few-shot train выборку в standard формате.

    Args:
        n_shots: 10 | 20 | 50
        seed: 42 | 123 | 456
        processed_dir: путь к data/processed/

    Returns:
        {"texts": list[str], "labels": list[int]}
    """
    fewshot_path = processed_dir / "clinc150_fewshot.json"
    with open(fewshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_key = f"n{n_shots}"
    seed_key = f"seed{seed}"
    return data[n_key][seed_key]["standard"]


def load_fewshot_autointent(n_shots: int, seed: int, processed_dir: Path) -> list[dict]:
    """
    Загружает few-shot train выборку в формате AutoIntent.

    Args:
        n_shots: 10 | 20 | 50
        seed: 42 | 123 | 456
        processed_dir: путь к data/processed/

    Returns:
        list[dict] — формат AutoIntent
    """
    fewshot_path = processed_dir / "clinc150_fewshot.json"
    with open(fewshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_key = f"n{n_shots}"
    seed_key = f"seed{seed}"
    return data[n_key][seed_key]["autointent"]


def load_intents(processed_dir: Path) -> list[dict]:
    """
    Загружает список in-scope интентов.

    Args:
        processed_dir: путь к data/processed/

    Returns:
        [{"id": int, "name": str}, ...] отсортировано по id
    """
    full_path = processed_dir / "clinc150_full.json"
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def load_meta(processed_dir: Path) -> dict:
    """
    Загружает метаданные датасета.

    Args:
        processed_dir: путь к data/processed/

    Returns:
        dict с метаданными (см. clinc150_meta.json)
    """
    meta_path = processed_dir / "clinc150_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_intent_names(processed_dir: Path) -> dict[int, str]:
    """
    Возвращает mapping label_id -> intent_name для in-scope интентов.

    Args:
        processed_dir: путь к data/processed/

    Returns:
        {label_id (int): intent_name (str)}
        Не включает OOS (label -1)
    """
    meta = load_meta(processed_dir)
    return {int(k): v for k, v in meta["intent_names"].items()}


# Legacy functions for compatibility (deprecated)

def sample_fewshot(
    data: dict,
    n_shots: int,
    oos_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    DEPRECATED: Use load_fewshot() instead.
    Few-shot sampling is now done in prepare_data.py.
    """
    raise NotImplementedError(
        "sample_fewshot is deprecated. "
        "Run `python scripts/prepare_data.py` first, "
        "then use `load_fewshot(n_shots, seed, processed_dir)`."
    )


def convert_to_autointent_format(data: dict) -> list[dict]:
    """
    DEPRECATED: Use load_clinc150_autointent() or load_fewshot_autointent().
    Conversion is now done in prepare_data.py.
    """
    raise NotImplementedError(
        "convert_to_autointent_format is deprecated. "
        "Run `python scripts/prepare_data.py` first, "
        "then use load_clinc150_autointent() or load_fewshot_autointent()."
    )
