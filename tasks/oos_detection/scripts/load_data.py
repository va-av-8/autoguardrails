#!/usr/bin/env python3
"""
CLI для загрузки данных CLINC150.

Источники данных:
    - standard: локальный файл data_full.json
    - autointent: HuggingFace DeepPavlov/clinc150

Примеры использования:
    # Загрузка из standard (data_full.json)
    python load_data.py --source standard --split train
    python load_data.py --source standard --split val --stats
    python load_data.py --source standard --split test --output data.json

    # Загрузка из autointent (HuggingFace)
    python load_data.py --source autointent --split train
    python load_data.py --source autointent --split validation --stats

    # Показать интенты
    python load_data.py --source standard --intents
    python load_data.py --source autointent --intents
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_utils import (
    load_standard,
    load_autointent,
    get_standard_intents,
    get_autointent_intents,
)


# Путь по умолчанию к data_full.json
DEFAULT_STANDARD_PATH = Path(__file__).parent.parent / "data/raw/data_full.json"


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка данных CLINC150",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--source",
        choices=["standard", "autointent"],
        required=True,
        help="Источник данных: standard (data_full.json) или autointent (HuggingFace)"
    )
    parser.add_argument(
        "--split",
        help="Сплит данных: train/val/test (standard) или train/validation/test (autointent)"
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Путь к data_full.json (только для standard)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Файл для сохранения результата"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Показать статистику вместо данных"
    )
    parser.add_argument(
        "--intents",
        action="store_true",
        help="Показать список интентов"
    )

    args = parser.parse_args()

    # Проверка аргументов
    if not args.intents and not args.split:
        parser.error("--split is required unless --intents is specified")

    # Вывод списка интентов
    if args.intents:
        if args.source == "standard":
            data_path = args.path or DEFAULT_STANDARD_PATH
            if not data_path.exists():
                print(f"Ошибка: файл не найден: {data_path}", file=sys.stderr)
                sys.exit(1)
            intents = get_standard_intents(data_path)
        else:
            intents = get_autointent_intents()

        output = {"intents": intents, "count": len(intents)}
        _output_json(output, args.output)
        return

    # Загрузка данных
    if args.source == "standard":
        data_path = args.path or DEFAULT_STANDARD_PATH
        if not data_path.exists():
            print(f"Ошибка: файл не найден: {data_path}", file=sys.stderr)
            sys.exit(1)

        valid_splits = ("train", "val", "test")
        if args.split not in valid_splits:
            print(f"Ошибка: для standard допустимые сплиты: {valid_splits}", file=sys.stderr)
            sys.exit(1)

        data = load_standard(data_path, args.split)
    else:
        valid_splits = ("train", "validation", "test")
        if args.split not in valid_splits:
            print(f"Ошибка: для autointent допустимые сплиты: {valid_splits}", file=sys.stderr)
            sys.exit(1)

        data = load_autointent(args.split)

    # Вывод статистики
    if args.stats:
        stats = _compute_stats(data)
        stats["source"] = args.source
        stats["split"] = args.split
        _output_json(stats, args.output)
        return

    # Вывод данных
    _output_json(data, args.output)


def _compute_stats(data: dict) -> dict:
    """Вычисляет статистику по данным."""
    total = len(data["texts"])
    n_oos = sum(1 for l in data["labels"] if l == -1)
    n_inscope = total - n_oos
    labels = set(l for l in data["labels"] if l != -1)

    return {
        "total": total,
        "n_inscope": n_inscope,
        "n_oos": n_oos,
        "n_unique_labels": len(labels),
        "oos_ratio": round(n_oos / total, 4) if total > 0 else 0
    }


def _output_json(data, output_path: Path | None):
    """Выводит данные в JSON."""
    result = json.dumps(data, ensure_ascii=False, indent=2)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Сохранено в {output_path}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
