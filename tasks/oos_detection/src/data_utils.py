"""
Утилиты для загрузки подготовленных данных CLINC150.

Данные должны быть предварительно созданы через:
    python scripts/prepare_data.py --source standard
    python scripts/prepare_data.py --source deeppavlov

Источники:
    - standard: github.com/clinc/oos-eval (100 OOS train)
    - deeppavlov: HuggingFace DeepPavlov/clinc150 (200 OOS train)
"""

from __future__ import annotations

import json
from pathlib import Path


# === Constants ===

OOS_LABEL = -1
VALID_SOURCES = ("standard", "deeppavlov")
VALID_SPLITS = ("train", "validation", "test")

# Default path to processed data
_DEFAULT_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _get_processed_dir(source: str, processed_dir: Path | None = None) -> Path:
    """Get path to processed data directory for a source."""
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got '{source}'")

    base_dir = processed_dir or _DEFAULT_PROCESSED_DIR
    source_dir = base_dir / source

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found: {source_dir}\n"
            f"Run: python scripts/prepare_data.py --source {source}"
        )

    return source_dir


# === Loading functions ===

def load_split(
    source: str,
    split: str,
    processed_dir: Path | None = None,
) -> dict:
    """
    Загружает сплит данных.

    Args:
        source: "standard" | "deeppavlov"
        split: "train" | "validation" | "test"
        processed_dir: путь к data/processed/ (опционально)

    Returns:
        {"texts": list[str], "labels": list[int]}
        OOS имеет label = -1
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

    source_dir = _get_processed_dir(source, processed_dir)
    full_path = source_dir / "full.json"

    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data[split]


def load_fewshot(
    source: str,
    n_shots: int,
    seed: int,
    processed_dir: Path | None = None,
) -> dict:
    """
    Загружает few-shot train выборку.

    Args:
        source: "standard" | "deeppavlov"
        n_shots: 10 | 20 | 50
        seed: 42 | 123 | 456
        processed_dir: путь к data/processed/ (опционально)

    Returns:
        {"texts": list[str], "labels": list[int]}
    """
    source_dir = _get_processed_dir(source, processed_dir)
    fewshot_path = source_dir / "fewshot.json"

    with open(fewshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_key = f"n{n_shots}"
    seed_key = f"seed{seed}"

    if n_key not in data:
        raise ValueError(f"n_shots={n_shots} not found. Available: {list(data.keys())}")
    if seed_key not in data[n_key]:
        raise ValueError(f"seed={seed} not found. Available: {list(data[n_key].keys())}")

    return data[n_key][seed_key]


def get_intents(
    source: str,
    processed_dir: Path | None = None,
) -> list[dict]:
    """
    Возвращает список интентов.

    Args:
        source: "standard" | "deeppavlov"
        processed_dir: путь к data/processed/ (опционально)

    Returns:
        [{"id": int, "name": str}, ...] отсортировано по id
    """
    source_dir = _get_processed_dir(source, processed_dir)
    meta_path = source_dir / "meta.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta["intents"]


def get_intent_names(
    source: str,
    processed_dir: Path | None = None,
) -> dict[int, str]:
    """
    Возвращает mapping label_id -> intent_name.

    Args:
        source: "standard" | "deeppavlov"
        processed_dir: путь к data/processed/ (опционально)

    Returns:
        {label_id: intent_name}
        Не включает OOS (label -1)
    """
    intents = get_intents(source, processed_dir)
    return {intent["id"]: intent["name"] for intent in intents}


def load_meta(
    source: str,
    processed_dir: Path | None = None,
) -> dict:
    """
    Загружает метаданные датасета.

    Args:
        source: "standard" | "deeppavlov"
        processed_dir: путь к data/processed/ (опционально)

    Returns:
        dict с метаданными
    """
    source_dir = _get_processed_dir(source, processed_dir)
    meta_path = source_dir / "meta.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_split_stats(
    source: str,
    processed_dir: Path | None = None,
) -> dict:
    """
    Возвращает статистику по сплитам.

    Args:
        source: "standard" | "deeppavlov"
        processed_dir: путь к data/processed/ (опционально)

    Returns:
        {split_name: {"total": int, "n_inscope": int, "n_oos": int}}
    """
    meta = load_meta(source, processed_dir)
    return meta["splits"]


# === AutoIntent format conversion ===

def to_autointent_format(data: dict) -> list[dict]:
    """
    Конвертирует standard формат в AutoIntent формат.

    Args:
        data: {"texts": list[str], "labels": list[int]} с OOS=-1

    Returns:
        list[dict] — in-scope: {"utterance": str, "label": int}
                     OOS: {"utterance": str} (без поля label)
    """
    result = []
    for text, label in zip(data["texts"], data["labels"]):
        if label == OOS_LABEL:
            result.append({"utterance": text})
        else:
            result.append({"utterance": text, "label": label})
    return result


def load_split_autointent(
    source: str,
    split: str,
    processed_dir: Path | None = None,
) -> list[dict]:
    """
    Загружает сплит в формате AutoIntent.

    Args:
        source: "standard" | "deeppavlov"
        split: "train" | "validation" | "test"
        processed_dir: путь к data/processed/ (опционально)

    Returns:
        list[dict] в формате AutoIntent
    """
    data = load_split(source, split, processed_dir)
    return to_autointent_format(data)


def load_fewshot_autointent(
    source: str,
    n_shots: int,
    seed: int,
    processed_dir: Path | None = None,
) -> list[dict]:
    """
    Загружает few-shot выборку в формате AutoIntent.

    Args:
        source: "standard" | "deeppavlov"
        n_shots: 10 | 20 | 50
        seed: 42 | 123 | 456
        processed_dir: путь к data/processed/ (опционально)

    Returns:
        list[dict] в формате AutoIntent
    """
    data = load_fewshot(source, n_shots, seed, processed_dir)
    return to_autointent_format(data)
