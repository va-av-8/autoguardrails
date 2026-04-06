"""
Подготовка данных WildJailbreak для экспериментов Jailbreak Detection.

Загружает данные из HuggingFace (allenai/wildjailbreak),
конвертирует в бинарный формат (safe/jailbreak),
создаёт few-shot сплиты в формате AutoIntent.

Использование:
    python scripts/prepare_data.py

    С кастомными параметрами:
    python scripts/prepare_data.py \
        --raw_train data/raw/wildjailbreak_train.jsonl \
        --raw_eval data/raw/wildjailbreak_eval.jsonl \
        --output_dir data/processed \
        --n_shots 10 20 50 \
        --seeds 42 123 456

Выходные файлы:
    data/raw/wildjailbreak_train.jsonl   — сырые train данные
    data/raw/wildjailbreak_eval.jsonl    — сырые eval данные
    data/processed/wildjailbreak_eval_binary.jsonl — eval с binary_label
    data/processed/train_shot{N}_seed{S}.json — few-shot сплиты (AutoIntent)
    data/processed/test.json             — тестовый сплит (AutoIntent)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


# Constants
HARMFUL_TYPES = {"vanilla_harmful", "adversarial_harmful"}
BENIGN_TYPES = {"vanilla_benign", "adversarial_benign"}
DEFAULT_N_SHOTS = [10, 20, 50]
DEFAULT_SEEDS = [42, 123, 456]


def get_binary_label(data_type: str) -> str:
    """Convert data_type to binary label."""
    if data_type in HARMFUL_TYPES:
        return "jailbreak"
    elif data_type in BENIGN_TYPES:
        return "safe"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def get_prompt_train(example: dict) -> str:
    """
    Get prompt text from train example.
    Train has columns: vanilla, adversarial, completion, data_type.
    If adversarial is non-empty, use it; otherwise use vanilla.
    """
    adversarial = example.get("adversarial", "")
    if adversarial and adversarial.strip():
        return adversarial
    return example["vanilla"]


def get_prompt_eval(example: dict) -> str:
    """
    Get prompt text from eval example.
    Eval has columns: adversarial, label, data_type.
    The 'adversarial' column contains the prompt.
    """
    return example["adversarial"]


def load_and_save_raw(output_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Load WildJailbreak from HuggingFace and save raw data.

    Returns:
        (train_data, eval_data) as lists of dicts
    """
    raw_dir = output_dir.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WildJailbreak from HuggingFace...")

    # Load train split (has columns: vanilla, adversarial, completion, data_type)
    print("Loading train split...")
    train_dataset = load_dataset(
        "allenai/wildjailbreak",
        data_files={"train": "train/train.tsv"},
        split="train",
        delimiter="\t",
        keep_default_na=False
    )
    train_data = list(train_dataset)
    print(f"Train loaded: {len(train_data)} examples")

    # Load eval split separately (has columns: adversarial, label, data_type)
    # Note: eval split has different schema - adversarial contains the prompt
    print("Loading eval split...")
    eval_dataset = load_dataset(
        "allenai/wildjailbreak",
        data_files={"test": "eval/eval.tsv"},
        split="test",
        delimiter="\t",
        keep_default_na=False
    )
    eval_data = list(eval_dataset)
    print(f"Eval loaded: {len(eval_data)} examples")

    # Save raw data
    train_path = raw_dir / "wildjailbreak_train.jsonl"
    eval_path = raw_dir / "wildjailbreak_eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved raw train: {train_path} ({len(train_data)} examples)")
    print(f"Saved raw eval:  {eval_path} ({len(eval_data)} examples)")

    return train_data, eval_data


def load_raw_from_files(raw_train: Path, raw_eval: Path) -> tuple[list[dict], list[dict]]:
    """Load raw data from existing JSONL files."""
    train_data = []
    with open(raw_train, "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))

    eval_data = []
    with open(raw_eval, "r", encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line))

    return train_data, eval_data


def process_eval_binary(eval_data: list[dict], output_dir: Path) -> dict:
    """
    Process eval data: add binary_label and prompt columns.
    Save to wildjailbreak_eval_binary.jsonl.

    Returns:
        Distribution statistics
    """
    processed = []
    stats = {"data_type": {}, "binary_label": {"safe": 0, "jailbreak": 0}}

    for example in eval_data:
        data_type = example["data_type"]
        binary_label = get_binary_label(data_type)
        prompt = get_prompt_eval(example)

        processed.append({
            **example,
            "binary_label": binary_label,
            "prompt": prompt
        })

        # Update stats
        stats["data_type"][data_type] = stats["data_type"].get(data_type, 0) + 1
        stats["binary_label"][binary_label] += 1

    # Save
    output_path = output_dir / "wildjailbreak_eval_binary.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved eval with binary labels: {output_path}")

    return stats


def create_fewshot_split(
    train_data: list[dict],
    n_shots: int,
    seed: int,
    output_dir: Path
) -> None:
    """
    Create stratified few-shot split in AutoIntent format.

    AutoIntent format:
    {
        "intents": [{"id": 0, "name": "safe", "utterances": [...]}],
        "oos_utterances": [...]
    }
    """
    np.random.seed(seed)

    # Group by binary label
    safe_examples = []
    jailbreak_examples = []

    for example in train_data:
        prompt = get_prompt_train(example)
        binary_label = get_binary_label(example["data_type"])

        if binary_label == "safe":
            safe_examples.append(prompt)
        else:
            jailbreak_examples.append(prompt)

    # Sample n_shots per class
    safe_sampled = list(np.random.choice(
        safe_examples,
        size=min(n_shots, len(safe_examples)),
        replace=False
    ))
    jailbreak_sampled = list(np.random.choice(
        jailbreak_examples,
        size=min(n_shots, len(jailbreak_examples)),
        replace=False
    ))

    # Create AutoIntent format
    autointent_data = {
        "intents": [
            {
                "id": 0,
                "name": "safe",
                "utterances": safe_sampled
            }
        ],
        "oos_utterances": jailbreak_sampled
    }

    # Save
    output_path = output_dir / f"train_shot{n_shots}_seed{seed}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(autointent_data, f, ensure_ascii=False, indent=2)


def create_test_split(eval_data: list[dict], output_dir: Path) -> None:
    """
    Create test split in AutoIntent format.

    Format:
    {
        "utterances": [...],
        "labels": [...]  # 0 = safe, 1 = jailbreak/OOS
    }
    """
    utterances = []
    labels = []

    for example in eval_data:
        prompt = get_prompt_eval(example)
        binary_label = get_binary_label(example["data_type"])

        utterances.append(prompt)
        labels.append(0 if binary_label == "safe" else 1)

    test_data = {
        "utterances": utterances,
        "labels": labels
    }

    # Save
    output_path = output_dir / "test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Saved test split: {output_path}")


def print_distribution(stats: dict) -> None:
    """Print class distribution."""
    print("\n=== Eval Distribution ===")
    print("\nBy data_type:")
    for dtype, count in sorted(stats["data_type"].items()):
        print(f"  {dtype}: {count}")

    print("\nBy binary_label:")
    for label, count in sorted(stats["binary_label"].items()):
        print(f"  {label}: {count}")

    total = sum(stats["binary_label"].values())
    print(f"\nTotal: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WildJailbreak data for jailbreak detection experiments"
    )
    parser.add_argument(
        "--raw_train",
        type=Path,
        help="Path to existing raw train JSONL (skip download if provided)"
    )
    parser.add_argument(
        "--raw_eval",
        type=Path,
        help="Path to existing raw eval JSONL (skip download if provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        nargs="+",
        default=DEFAULT_N_SHOTS,
        help="Number of shots per class for few-shot splits"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds for few-shot sampling"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.raw_train and args.raw_eval:
        print(f"Loading from existing files...")
        train_data, eval_data = load_raw_from_files(args.raw_train, args.raw_eval)
        print(f"Loaded train: {len(train_data)} examples")
        print(f"Loaded eval:  {len(eval_data)} examples")
    else:
        train_data, eval_data = load_and_save_raw(output_dir)

    # Process eval with binary labels
    stats = process_eval_binary(eval_data, output_dir)
    print_distribution(stats)

    # Create few-shot splits
    print("\n=== Creating Few-Shot Splits ===")
    for n_shots in args.n_shots:
        for seed in args.seeds:
            create_fewshot_split(train_data, n_shots, seed, output_dir)
            print(f"Created train_shot{n_shots}_seed{seed}.json")

    # Create test split
    print("\n=== Creating Test Split ===")
    create_test_split(eval_data, output_dir)

    print(f"\nDone! All files saved to {output_dir}/")


if __name__ == "__main__":
    main()
