#!/usr/bin/env python3
"""
Подготовка данных CLINC150 для экспериментов.

Источники:
    - standard: github.com/clinc/oos-eval (100 OOS train, для воспроизводимости ADB/DETER)
    - deeppavlov: HuggingFace DeepPavlov/clinc150 (200 OOS train, для AutoIntent)

Использование:
    python prepare_data.py --source standard
    python prepare_data.py --source deeppavlov
    python prepare_data.py --source all

Выходные файлы:
    data/processed/{source}/full.json     — полные сплиты
    data/processed/{source}/fewshot.json  — few-shot выборки
    data/processed/{source}/meta.json     — метаданные
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np


# === Constants ===

OOS_LABEL = -1
N_INTENTS = 150
N_SHOTS_LIST = [10, 20, 50]
SEEDS = [42, 123, 456]
OOS_RATIO = 0.1

STANDARD_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# === Download functions ===

def download_standard() -> dict:
    """Download data from github.com/clinc/oos-eval."""
    cache_path = RAW_DIR / "standard_data_full.json"

    if cache_path.exists():
        print(f"Using cached: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Downloading from {STANDARD_URL}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(STANDARD_URL) as response:
        data = json.loads(response.read().decode("utf-8"))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Saved to {cache_path}")
    return data


def download_deeppavlov() -> dict:
    """Download data from HuggingFace DeepPavlov/clinc150."""
    from datasets import load_dataset

    print("Loading DeepPavlov/clinc150 from HuggingFace...")
    ds = load_dataset("DeepPavlov/clinc150")

    # Use numeric IDs as intent names (0-149)
    intent_names = {i: f"intent_{i}" for i in range(N_INTENTS)}

    return {"dataset": ds, "intent_names": intent_names}


# === Processing functions ===

def process_standard(raw_data: dict) -> tuple[dict, dict, list[dict]]:
    """
    Process standard format data.

    Input format: {"train": [[text, label], ...], "oos_train": [[text, "oos"], ...], ...}
    Output format: {"texts": [...], "labels": [...]} with OOS=-1
    """
    # Collect all intent names and create mapping
    all_intents = set()
    for split in ["train", "val", "test"]:
        for text, label in raw_data[split]:
            all_intents.add(label)

    intent_to_id = {name: idx for idx, name in enumerate(sorted(all_intents))}
    intents_list = [{"id": idx, "name": name} for name, idx in sorted(intent_to_id.items(), key=lambda x: x[1])]

    # Process splits
    splits = {}
    split_mapping = {"train": "train", "val": "validation", "test": "test"}

    for src_split, dst_split in split_mapping.items():
        texts = []
        labels = []
        seen = set()

        # In-scope examples
        for text, label in raw_data[src_split]:
            if text not in seen:
                seen.add(text)
                texts.append(text)
                labels.append(intent_to_id[label])

        # OOS examples
        oos_key = f"oos_{src_split}"
        if oos_key in raw_data:
            for text, _ in raw_data[oos_key]:
                if text not in seen:
                    seen.add(text)
                    texts.append(text)
                    labels.append(OOS_LABEL)

        splits[dst_split] = {"texts": texts, "labels": labels}

    # Stats
    stats = {}
    for split_name, data in splits.items():
        n_oos = sum(1 for l in data["labels"] if l == OOS_LABEL)
        stats[split_name] = {
            "total": len(data["texts"]),
            "n_inscope": len(data["texts"]) - n_oos,
            "n_oos": n_oos
        }

    return splits, stats, intents_list


def process_deeppavlov(raw_data: dict) -> tuple[dict, dict, list[dict]]:
    """
    Process DeepPavlov/clinc150 data.

    OOS examples have label=None in this dataset.
    """
    ds = raw_data["dataset"]
    intent_names = raw_data["intent_names"]

    intents_list = [{"id": i, "name": intent_names[i]} for i in sorted(intent_names.keys())]

    splits = {}
    stats = {}

    for split_name in ["train", "validation", "test"]:
        texts = []
        labels = []
        seen = set()

        for item in ds[split_name]:
            text = item["utterance"]
            if text not in seen:
                seen.add(text)
                texts.append(text)
                if item["label"] is None:
                    labels.append(OOS_LABEL)
                else:
                    labels.append(item["label"])

        splits[split_name] = {"texts": texts, "labels": labels}

        n_oos = sum(1 for l in labels if l == OOS_LABEL)
        stats[split_name] = {
            "total": len(texts),
            "n_inscope": len(texts) - n_oos,
            "n_oos": n_oos
        }

    return splits, stats, intents_list


def create_fewshot_splits(train_data: dict) -> dict:
    """Create few-shot training splits."""
    texts = train_data["texts"]
    labels = train_data["labels"]

    # Group by label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    fewshot = {}

    for n_shots in N_SHOTS_LIST:
        n_key = f"n{n_shots}"
        fewshot[n_key] = {}

        for seed in SEEDS:
            seed_key = f"seed{seed}"
            np.random.seed(seed)

            sampled_texts = []
            sampled_labels = []

            # Sample n_shots per in-scope intent
            for label in sorted(label_to_indices.keys()):
                if label == OOS_LABEL:
                    continue
                indices = label_to_indices[label]
                n_sample = min(n_shots, len(indices))
                sampled_idx = np.random.choice(indices, size=n_sample, replace=False)
                for idx in sampled_idx:
                    sampled_texts.append(texts[idx])
                    sampled_labels.append(labels[idx])

            # Sample OOS proportionally
            n_oos = round(N_INTENTS * n_shots * OOS_RATIO)
            if OOS_LABEL in label_to_indices:
                oos_indices = label_to_indices[OOS_LABEL]
                n_oos_sample = min(n_oos, len(oos_indices))
                sampled_oos_idx = np.random.choice(oos_indices, size=n_oos_sample, replace=False)
                for idx in sampled_oos_idx:
                    sampled_texts.append(texts[idx])
                    sampled_labels.append(labels[idx])

            fewshot[n_key][seed_key] = {
                "texts": sampled_texts,
                "labels": sampled_labels
            }

    return fewshot


def save_processed(source: str, splits: dict, fewshot: dict, stats: dict, intents: list[dict]):
    """Save processed data to disk."""
    output_dir = PROCESSED_DIR / source
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full splits
    full_path = output_dir / "full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    # Few-shot splits
    fewshot_path = output_dir / "fewshot.json"
    with open(fewshot_path, "w", encoding="utf-8") as f:
        json.dump(fewshot, f, ensure_ascii=False, indent=2)

    # Metadata
    meta = {
        "source": source,
        "n_intents": N_INTENTS,
        "oos_label": OOS_LABEL,
        "splits": stats,
        "fewshot": {
            "n_shots": N_SHOTS_LIST,
            "seeds": SEEDS,
            "oos_ratio": OOS_RATIO
        },
        "intents": intents
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_dir}/")


def prepare_source(source: str):
    """Prepare data for a single source."""
    print(f"\n{'='*50}")
    print(f"Preparing: {source}")
    print("="*50)

    # Download
    if source == "standard":
        raw_data = download_standard()
        splits, stats, intents = process_standard(raw_data)
    else:  # deeppavlov
        raw_data = download_deeppavlov()
        splits, stats, intents = process_deeppavlov(raw_data)

    # Print stats
    for split_name, s in stats.items():
        print(f"{split_name:12}: {s['n_inscope']:5} in-scope + {s['n_oos']:4} OOS = {s['total']:5} total")

    # Create few-shot splits
    fewshot = create_fewshot_splits(splits["train"])
    print(f"\nFew-shot splits: n={N_SHOTS_LIST}, seeds={SEEDS}")

    # Save
    save_processed(source, splits, fewshot, stats, intents)


def main():
    parser = argparse.ArgumentParser(description="Prepare CLINC150 data")
    parser.add_argument(
        "--source",
        choices=["standard", "deeppavlov", "all"],
        default="all",
        help="Data source: standard (github/clinc), deeppavlov (HuggingFace), or all"
    )
    args = parser.parse_args()

    sources = ["standard", "deeppavlov"] if args.source == "all" else [args.source]

    for source in sources:
        prepare_source(source)

    print("\nDone!")


if __name__ == "__main__":
    main()
