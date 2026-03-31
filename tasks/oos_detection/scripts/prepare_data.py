"""
Подготовка данных CLINC150 для экспериментов.

Загружает данные из HuggingFace (clinc_oos, config "plus"),
конвертирует в standard и AutoIntent форматы,
сохраняет в data/processed/.

Использование:
    python scripts/prepare_data.py

Выходные файлы:
    data/processed/clinc150_full.json     — полные сплиты
    data/processed/clinc150_fewshot.json  — few-shot выборки
    data/processed/clinc150_meta.json     — метаданные
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


# Constants
OOS_LABEL_HF = 42        # OOS label in HuggingFace dataset
OOS_LABEL_STANDARD = -1  # OOS label in our standard format
N_INTENTS = 150          # Number of in-scope intents
N_SHOTS_LIST = [10, 20, 50]
SEEDS = [42, 123, 456]
OOS_RATIO = 0.1


def verify_oos_label(dataset) -> None:
    """Verify that label 42 is indeed 'oos'."""
    features = dataset["train"].features["intent"]
    intent_name = features.int2str(OOS_LABEL_HF)
    assert intent_name == "oos", f"Expected label 42 to be 'oos', got '{intent_name}'"


def get_intent_names(dataset) -> dict[int, str]:
    """Get mapping from label id to intent name."""
    features = dataset["train"].features["intent"]
    intent_names = {}
    for i in range(features.num_classes):
        intent_names[i] = features.int2str(i)
    return intent_names


def deduplicate_split(split_data, exclude_texts: set | None = None) -> list[dict]:
    """
    Deduplicate examples by text, keeping first occurrence.
    HuggingFace clinc_oos "plus" config now contains duplicates
    from multiple configurations.

    Args:
        split_data: HuggingFace dataset split
        exclude_texts: set of texts to exclude (e.g., from train when processing test)

    Returns:
        list of unique examples not in exclude_texts
    """
    if exclude_texts is None:
        exclude_texts = set()

    seen_texts = set()
    unique_examples = []
    for example in split_data:
        text = example["text"]
        if text not in seen_texts and text not in exclude_texts:
            seen_texts.add(text)
            unique_examples.append({"text": text, "intent": example["intent"]})
    return unique_examples


def convert_to_standard_format(examples: list[dict], label_mapping: dict[int, int]) -> dict:
    """
    Convert examples to standard format.
    OOS (intent 42) -> -1, in-scope labels remapped to sequential 0-149.
    """
    texts = []
    labels = []
    for example in examples:
        texts.append(example["text"])
        hf_label = example["intent"]
        if hf_label == OOS_LABEL_HF:
            labels.append(OOS_LABEL_STANDARD)
        else:
            labels.append(label_mapping[hf_label])
    return {"texts": texts, "labels": labels}


def convert_to_autointent_format(standard_data: dict) -> list[dict]:
    """
    Convert standard format to AutoIntent format.
    In-scope: {"utterance": text, "label": int}
    OOS: {"utterance": text} (no label field)
    """
    result = []
    for text, label in zip(standard_data["texts"], standard_data["labels"]):
        if label == OOS_LABEL_STANDARD:
            result.append({"utterance": text})
        else:
            result.append({"utterance": text, "label": label})
    return result


def build_label_mapping(intent_names: dict[int, str]) -> dict[int, int]:
    """
    Build mapping from HF label IDs to sequential 0-149 IDs.

    HuggingFace clinc_oos uses label 42 for OOS, so in-scope labels are:
    0-41, 43-150 (150 total). We remap these to sequential 0-149.

    Returns:
        {hf_label: new_label} for in-scope intents only
    """
    mapping = {}
    new_id = 0
    for hf_id in sorted(intent_names.keys()):
        if hf_id != OOS_LABEL_HF:  # Skip OOS
            mapping[hf_id] = new_id
            new_id += 1
    assert new_id == N_INTENTS, f"Expected {N_INTENTS} intents, got {new_id}"
    return mapping


def get_intents_list(intent_names: dict[int, str], label_mapping: dict[int, int]) -> list[dict]:
    """Get list of in-scope intents with sequential IDs 0-149."""
    intents = []
    for hf_id, new_id in sorted(label_mapping.items(), key=lambda x: x[1]):
        intents.append({"id": new_id, "name": intent_names[hf_id]})
    return intents


def sample_fewshot(
    standard_data: dict,
    n_shots: int,
    seed: int,
) -> dict:
    """
    Sample n examples per in-scope intent + proportional OOS.

    Args:
        standard_data: {"texts": [...], "labels": [...]} with -1 for OOS
        n_shots: number of examples per in-scope intent
        seed: random seed for reproducibility

    Returns:
        dict in same format with sampled data
    """
    np.random.seed(seed)

    texts = standard_data["texts"]
    labels = standard_data["labels"]

    # Group examples by label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    sampled_texts = []
    sampled_labels = []

    # Sample n_shots per in-scope intent
    for label in sorted(label_to_indices.keys()):
        if label == OOS_LABEL_STANDARD:
            continue  # Handle OOS separately
        indices = label_to_indices[label]
        if len(indices) < n_shots:
            # If not enough examples, take all
            sampled_idx = indices
        else:
            sampled_idx = np.random.choice(indices, size=n_shots, replace=False)
        for idx in sampled_idx:
            sampled_texts.append(texts[idx])
            sampled_labels.append(labels[idx])

    # Sample OOS: round(n_intents * n_shots * oos_ratio)
    n_oos = round(N_INTENTS * n_shots * OOS_RATIO)
    if OOS_LABEL_STANDARD in label_to_indices:
        oos_indices = label_to_indices[OOS_LABEL_STANDARD]
        if len(oos_indices) >= n_oos:
            sampled_oos_idx = np.random.choice(oos_indices, size=n_oos, replace=False)
        else:
            sampled_oos_idx = oos_indices  # Take all available
        for idx in sampled_oos_idx:
            sampled_texts.append(texts[idx])
            sampled_labels.append(labels[idx])

    return {"texts": sampled_texts, "labels": sampled_labels}


def count_split_stats(standard_data: dict) -> dict:
    """Count in-scope and OOS examples in a split."""
    labels = standard_data["labels"]
    n_oos = sum(1 for l in labels if l == OOS_LABEL_STANDARD)
    n_inscope = len(labels) - n_oos
    return {
        "total": len(labels),
        "n_inscope": n_inscope,
        "n_oos": n_oos
    }


def main():
    # Determine output directory relative to script location
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading clinc_oos (plus)...")
    dataset = load_dataset("clinc_oos", "plus", verification_mode="no_checks")

    # Verify OOS label
    verify_oos_label(dataset)
    print(f'OOS label: {OOS_LABEL_HF} ("oos") — verified')

    # Get intent names and build label mapping (HF IDs -> sequential 0-149)
    intent_names = get_intent_names(dataset)
    label_mapping = build_label_mapping(intent_names)
    print(f"Label mapping: {N_INTENTS} in-scope intents remapped to 0-{N_INTENTS-1}")

    # Convert all splits to standard and autointent formats
    # Note: deduplicate because HuggingFace now returns combined configs
    # Also remove train-test overlap by excluding train texts from val/test
    full_data = {
        "standard": {},
        "autointent": {},
        "intents": get_intents_list(intent_names, label_mapping)
    }

    split_stats = {}

    # Process train first (no exclusions)
    train_examples = deduplicate_split(dataset["train"])
    train_texts = {ex["text"] for ex in train_examples}
    train_standard = convert_to_standard_format(train_examples, label_mapping)
    full_data["standard"]["train"] = train_standard
    full_data["autointent"]["train"] = convert_to_autointent_format(train_standard)
    split_stats["train"] = count_split_stats(train_standard)

    # Process validation and test, excluding any texts from train
    for split_name in ["validation", "test"]:
        unique_examples = deduplicate_split(dataset[split_name], exclude_texts=train_texts)

        # Convert to standard format
        standard = convert_to_standard_format(unique_examples, label_mapping)
        autointent = convert_to_autointent_format(standard)
        full_data["standard"][split_name] = standard
        full_data["autointent"][split_name] = autointent
        split_stats[split_name] = count_split_stats(standard)

    # Print split stats
    print(f"Train:      {split_stats['train']['total']:5} total "
          f"({split_stats['train']['n_inscope']} in-scope + {split_stats['train']['n_oos']} OOS)")
    print(f"Validation: {split_stats['validation']['total']:5} total "
          f"({split_stats['validation']['n_inscope']} in-scope + {split_stats['validation']['n_oos']} OOS)")
    print(f"Test:       {split_stats['test']['total']:5} total "
          f"({split_stats['test']['n_inscope']} in-scope + {split_stats['test']['n_oos']} OOS)")

    # Save full data
    full_path = output_dir / "clinc150_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    # Create few-shot splits
    fewshot_data = {}
    train_standard = full_data["standard"]["train"]

    print("Few-shot splits:")
    for n_shots in N_SHOTS_LIST:
        n_key = f"n{n_shots}"
        fewshot_data[n_key] = {}
        n_oos_expected = round(N_INTENTS * n_shots * OOS_RATIO)

        for seed in SEEDS:
            seed_key = f"seed{seed}"
            fs_standard = sample_fewshot(train_standard, n_shots, seed)
            fs_autointent = convert_to_autointent_format(fs_standard)
            fewshot_data[n_key][seed_key] = {
                "standard": fs_standard,
                "autointent": fs_autointent
            }

        n_inscope = N_INTENTS * n_shots
        print(f"  n={n_shots:2}: {n_inscope} in-scope + {n_oos_expected} OOS per seed")

    # Save few-shot data
    fewshot_path = output_dir / "clinc150_fewshot.json"
    with open(fewshot_path, "w", encoding="utf-8") as f:
        json.dump(fewshot_data, f, ensure_ascii=False, indent=2)

    # Create metadata
    meta = {
        "dataset": "clinc_oos",
        "config": "plus",
        "hf_dataset_id": "clinc_oos",
        "source_paper": "Larson et al., EMNLP 2019",
        "source_repo": "https://github.com/clinc/oos-eval",
        "reproducibility_note": (
            "ADB/DETER use data_full.json (100 OOS train). "
            "We use HF 'plus' (250 OOS train). "
            "Test split is identical (5500 examples) so test metrics are comparable."
        ),
        "oos_label_hf": OOS_LABEL_HF,
        "oos_label_name": "oos",
        "oos_label_standard": OOS_LABEL_STANDARD,
        "n_intents": N_INTENTS,
        "splits": split_stats,
        "fewshot": {
            "n_shots": N_SHOTS_LIST,
            "seeds": SEEDS,
            "oos_ratio": OOS_RATIO,
            "oos_per_n": {
                f"n{n}": round(N_INTENTS * n * OOS_RATIO)
                for n in N_SHOTS_LIST
            }
        },
        "intent_names": {
            str(label_mapping[hf_id]): name
            for hf_id, name in intent_names.items()
            if hf_id != OOS_LABEL_HF  # Exclude OOS from intent names
        }
    }

    # Save metadata
    meta_path = output_dir / "clinc150_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
