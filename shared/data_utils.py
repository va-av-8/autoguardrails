"""
Утилиты для загрузки данных CLINC150.

Два источника данных:
    - standard: локальный файл data_full.json (формат testing_models_auto_intent)
    - autointent: HuggingFace DeepPavlov/clinc150
"""

from __future__ import annotations

import json
from pathlib import Path


# === Константы ===

OOS_LABEL_STANDARD = -1
OOS_LABEL_STRING = "oos"


# === Загрузка из standard формата (data_full.json) ===

def load_standard(
    json_path: Path,
    split: str = "train",
) -> dict:
    """
    Загружает данные из standard формата (data_full.json).

    Формат входного файла:
        {
            "train": [["text", "label"], ...],
            "val": [["text", "label"], ...],
            "test": [["text", "label"], ...],
            "oos_train": [["text", "oos"], ...],
            "oos_val": [["text", "oos"], ...],
            "oos_test": [["text", "oos"], ...]
        }

    Args:
        json_path: путь к data_full.json
        split: "train" | "val" | "test"

    Returns:
        {"texts": list[str], "labels": list[int]}
        OOS имеет label = -1
    """
    valid_splits = ("train", "val", "test")
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Собираем уникальные метки для создания маппинга
    all_labels = set()
    for item in raw[split]:
        if item[1] != OOS_LABEL_STRING:
            all_labels.add(item[1])

    # Создаём маппинг: label_name -> label_id (сортируем для воспроизводимости)
    label_to_id = {name: idx for idx, name in enumerate(sorted(all_labels))}

    texts = []
    labels = []

    # In-domain примеры
    for text, label in raw[split]:
        texts.append(text)
        labels.append(label_to_id[label])

    # OOS примеры
    oos_key = f"oos_{split}"
    if oos_key in raw:
        for text, _ in raw[oos_key]:
            texts.append(text)
            labels.append(OOS_LABEL_STANDARD)

    return {"texts": texts, "labels": labels}


def get_standard_intents(json_path: Path) -> list[dict]:
    """
    Извлекает список интентов из data_full.json.

    Args:
        json_path: путь к data_full.json

    Returns:
        [{"id": int, "name": str}, ...] отсортировано по id
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_labels = set()
    for split in ("train", "val", "test"):
        for item in raw[split]:
            if item[1] != OOS_LABEL_STRING:
                all_labels.add(item[1])

    sorted_labels = sorted(all_labels)
    return [{"id": idx, "name": name} for idx, name in enumerate(sorted_labels)]


# === Загрузка из HuggingFace DeepPavlov/clinc150 (autointent) ===

def load_autointent(
    split: str = "train",
) -> dict:
    """
    Загружает данные из HuggingFace DeepPavlov/clinc150.

    OOS примеры в этом датасете имеют label=None.

    Args:
        split: "train" | "validation" | "test"

    Returns:
        {"texts": list[str], "labels": list[int]}
        OOS имеет label = -1
    """
    from datasets import load_dataset

    valid_splits = ("train", "validation", "test")
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

    ds = load_dataset("DeepPavlov/clinc150", split=split)

    texts = []
    labels = []

    for item in ds:
        texts.append(item["utterance"])
        if item["label"] is None:
            labels.append(OOS_LABEL_STANDARD)
        else:
            labels.append(item["label"])

    return {"texts": texts, "labels": labels}


def get_autointent_intents() -> list[dict]:
    """
    Загружает список интентов из DeepPavlov/clinc150 (конфигурация intents).

    Returns:
        [{"id": int, "name": str}, ...]
    """
    from datasets import load_dataset

    ds = load_dataset("DeepPavlov/clinc150", "intents", split="intents")

    intents = []
    for item in ds:
        intents.append({"id": item["id"], "name": item["name"]})

    return sorted(intents, key=lambda x: x["id"])
