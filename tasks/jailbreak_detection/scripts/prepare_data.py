"""
Подготовка данных WildJailbreak для экспериментов Jailbreak Detection.

Загружает данные из HuggingFace (allenai/wildjailbreak),
конвертирует в бинарный формат (safe/jailbreak),
создаёт few-shot сплиты и full-train подвыборки в формате AutoIntent.

Использование:
    # Few-shot сплиты (по умолчанию)
    python scripts/prepare_data.py

    # Full-train 100K подвыборки для AutoML
    python scripts/prepare_data.py --full_subset

    Датасет allenai/wildjailbreak на Hugging Face закрытый (gated): на странице датасета
    нужно принять условия, затем авторизация — huggingface-cli login или переменная HF_TOKEN.

    # С кастомными параметрами
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
    data/processed/wildjailbreak_full100k_seed{S}.json — full-train 100K (AutoIntent)
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError


# Constants
HARMFUL_TYPES = {"vanilla_harmful", "adversarial_harmful"}
BENIGN_TYPES = {"vanilla_benign", "adversarial_benign"}
VANILLA_TYPES = {"vanilla_harmful", "vanilla_benign"}
ADVERSARIAL_TYPES = {"adversarial_harmful", "adversarial_benign"}
DEFAULT_N_SHOTS = [10, 20, 50]
DEFAULT_SEEDS = [42, 123, 456]


def _hf_token() -> str | None:
    """Токен для gated-датасетов (HF_TOKEN или HUGGING_FACE_HUB_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _print_hf_gated_help() -> None:
    print("\n" + "=" * 60)
    print("WildJailbreak (allenai/wildjailbreak) — gated dataset на Hugging Face.")
    print("1) Откройте страницу датасета и примите условия доступа.")
    print("2) Токен: https://huggingface.co/settings/tokens")
    print("3) Выполните:  huggingface-cli login")
    print("   или:        export HF_TOKEN=<ваш_токен>")
    print("=" * 60 + "\n")


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

    dl_mode = os.environ.get("HF_DATASETS_DOWNLOAD_MODE")  # напр. force_redownload
    _ld_extra: dict = {"token": _hf_token()}
    if dl_mode:
        _ld_extra["download_mode"] = dl_mode

    # Load train split (has columns: vanilla, adversarial, completion, data_type)
    print("Loading train split...")
    try:
        train_dataset = load_dataset(
            "allenai/wildjailbreak",
            data_files={"train": "train/train.tsv"},
            split="train",
            delimiter="\t",
            keep_default_na=False,
            **_ld_extra,
        )
    except DatasetNotFoundError:
        _print_hf_gated_help()
        raise

    train_data = list(train_dataset)
    print(f"Train loaded: {len(train_data)} examples")

    # Load eval split separately (has columns: adversarial, label, data_type)
    # Note: eval split has different schema - adversarial contains the prompt
    print("Loading eval split...")
    try:
        eval_dataset = load_dataset(
            "allenai/wildjailbreak",
            data_files={"test": "eval/eval.tsv"},
            split="test",
            delimiter="\t",
            keep_default_na=False,
            **_ld_extra,
        )
    except DatasetNotFoundError:
        _print_hf_gated_help()
        raise
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


def make_full_train_subset(
    train_data: list[dict],
    n_total: int = 100_000,
    seed: int = 42
) -> list[dict]:
    """
    Create stratified subset of train data for full-train AutoML experiments.

    Stratification:
    - 50/50 balance between safe and jailbreak classes
    - Within each class, preserve original vanilla/adversarial ratio

    Reference: Shi et al. 2021 (NeurIPS D&B) downsampled large text datasets
    (jigsaw toxicity, mercari) to 100K for AutoML computational tractability.

    Args:
        train_data: Full train dataset as list of dicts
        n_total: Target total size (default 100K)
        seed: Random seed for reproducibility

    Returns:
        Stratified subset of train_data
    """
    rng = np.random.default_rng(seed)

    # Group examples by stratum (data_type)
    strata = {
        "vanilla_harmful": [],
        "adversarial_harmful": [],
        "vanilla_benign": [],
        "adversarial_benign": [],
    }

    for idx, example in enumerate(train_data):
        data_type = example["data_type"]
        strata[data_type].append(idx)

    # Target: 50K per class
    n_per_class = n_total // 2

    # Calculate proportions within each class
    # Jailbreak class (harmful)
    n_vanilla_harmful = len(strata["vanilla_harmful"])
    n_adversarial_harmful = len(strata["adversarial_harmful"])
    n_harmful_total = n_vanilla_harmful + n_adversarial_harmful

    prop_vanilla_harmful = n_vanilla_harmful / n_harmful_total
    prop_adversarial_harmful = n_adversarial_harmful / n_harmful_total

    # Safe class (benign)
    n_vanilla_benign = len(strata["vanilla_benign"])
    n_adversarial_benign = len(strata["adversarial_benign"])
    n_benign_total = n_vanilla_benign + n_adversarial_benign

    prop_vanilla_benign = n_vanilla_benign / n_benign_total
    prop_adversarial_benign = n_adversarial_benign / n_benign_total

    # Calculate target counts per stratum
    targets = {
        "vanilla_harmful": int(round(n_per_class * prop_vanilla_harmful)),
        "adversarial_harmful": int(round(n_per_class * prop_adversarial_harmful)),
        "vanilla_benign": int(round(n_per_class * prop_vanilla_benign)),
        "adversarial_benign": int(round(n_per_class * prop_adversarial_benign)),
    }

    # Adjust for rounding to hit exact n_per_class per binary class
    harmful_sum = targets["vanilla_harmful"] + targets["adversarial_harmful"]
    if harmful_sum != n_per_class:
        targets["adversarial_harmful"] += (n_per_class - harmful_sum)

    benign_sum = targets["vanilla_benign"] + targets["adversarial_benign"]
    if benign_sum != n_per_class:
        targets["adversarial_benign"] += (n_per_class - benign_sum)

    # Sample from each stratum
    sampled_indices = []
    for stratum_name, target_n in targets.items():
        available = strata[stratum_name]
        actual_n = min(target_n, len(available))

        if actual_n < target_n:
            warnings.warn(
                f"Stratum '{stratum_name}' has only {len(available)} examples, "
                f"requested {target_n}. Using all available."
            )

        chosen = rng.choice(available, size=actual_n, replace=False)
        sampled_indices.extend(chosen)

    # Shuffle the combined sample
    rng.shuffle(sampled_indices)

    # Extract sampled examples
    sampled_data = [train_data[i] for i in sampled_indices]

    return sampled_data


def create_full_train_split(
    train_data: list[dict],
    n_total: int,
    seed: int,
    output_dir: Path
) -> dict:
    """
    Create stratified full-train subset in AutoIntent format.

    Args:
        train_data: Full train dataset
        n_total: Target total size
        seed: Random seed
        output_dir: Output directory

    Returns:
        Statistics dict for verification
    """
    # Get stratified subset
    subset = make_full_train_subset(train_data, n_total=n_total, seed=seed)

    # Convert to AutoIntent format
    safe_utterances = []
    jailbreak_utterances = []
    stats = {
        "total": len(subset),
        "safe": 0,
        "jailbreak": 0,
        "vanilla_harmful": 0,
        "adversarial_harmful": 0,
        "vanilla_benign": 0,
        "adversarial_benign": 0,
    }

    for example in subset:
        prompt = get_prompt_train(example)
        data_type = example["data_type"]
        binary_label = get_binary_label(data_type)

        stats[data_type] += 1

        if binary_label == "safe":
            safe_utterances.append(prompt)
            stats["safe"] += 1
        else:
            jailbreak_utterances.append(prompt)
            stats["jailbreak"] += 1

    # Create AutoIntent format (same as few-shot)
    autointent_data = {
        "intents": [
            {
                "id": 0,
                "name": "safe",
                "utterances": safe_utterances
            }
        ],
        "oos_utterances": jailbreak_utterances
    }

    # Save
    output_path = output_dir / f"wildjailbreak_full100k_seed{seed}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(autointent_data, f, ensure_ascii=False, indent=2)

    return stats


def verify_full_train_subsets(output_dir: Path, seeds: list[int]) -> None:
    """
    Verify full-train subsets: size, balance, and cross-seed overlap.

    Args:
        output_dir: Directory containing the subset files
        seeds: List of seeds used
    """
    print("\n" + "=" * 60)
    print("VERIFICATION: Full-Train 100K Subsets")
    print("=" * 60)

    all_indices = {}

    for seed in seeds:
        path = output_dir / f"wildjailbreak_full100k_seed{seed}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        safe = data["intents"][0]["utterances"]
        jailbreak = data["oos_utterances"]
        total = len(safe) + len(jailbreak)

        # Store utterances for overlap calculation
        all_indices[seed] = set(safe + jailbreak)

        print(f"\n--- Seed {seed} ---")
        print(f"Total: {total:,}")
        print(f"Safe: {len(safe):,} ({100*len(safe)/total:.1f}%)")
        print(f"Jailbreak: {len(jailbreak):,} ({100*len(jailbreak)/total:.1f}%)")

    # Calculate pairwise overlap
    print("\n--- Cross-seed Overlap ---")
    print("(Expected ~38K based on 100K×100K/261K)")
    seed_list = list(seeds)
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            s1, s2 = seed_list[i], seed_list[j]
            overlap = len(all_indices[s1] & all_indices[s2])
            print(f"Seed {s1} ∩ Seed {s2}: {overlap:,} utterances")

    print("=" * 60)


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
        help="Random seeds for sampling"
    )
    parser.add_argument(
        "--full_subset",
        action="store_true",
        help="Create 100K stratified full-train subsets for AutoML experiments"
    )
    parser.add_argument(
        "--full_subset_size",
        type=int,
        default=100_000,
        help="Size of full-train subset (default: 100000)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine raw data paths
    script_dir = Path(__file__).parent
    default_raw_dir = script_dir.parent / "data" / "raw"
    raw_train = args.raw_train or default_raw_dir / "wildjailbreak_train.jsonl"
    raw_eval = args.raw_eval or default_raw_dir / "wildjailbreak_eval.jsonl"

    # Load data
    if raw_train.exists() and raw_eval.exists():
        print(f"Loading from existing files...")
        train_data, eval_data = load_raw_from_files(raw_train, raw_eval)
        print(f"Loaded train: {len(train_data)} examples")
        print(f"Loaded eval:  {len(eval_data)} examples")
    else:
        train_data, eval_data = load_and_save_raw(output_dir)

    if args.full_subset:
        # Create full-train 100K subsets only
        print("\n=== Creating Full-Train 100K Subsets ===")
        print(f"Target size: {args.full_subset_size:,}")
        print(f"Seeds: {args.seeds}")

        for seed in args.seeds:
            stats = create_full_train_split(
                train_data,
                n_total=args.full_subset_size,
                seed=seed,
                output_dir=output_dir
            )
            print(f"\nCreated wildjailbreak_full100k_seed{seed}.json")
            print(f"  Total: {stats['total']:,}")
            print(f"  Safe: {stats['safe']:,} "
                  f"(vanilla: {stats['vanilla_benign']:,}, "
                  f"adversarial: {stats['adversarial_benign']:,})")
            print(f"  Jailbreak: {stats['jailbreak']:,} "
                  f"(vanilla: {stats['vanilla_harmful']:,}, "
                  f"adversarial: {stats['adversarial_harmful']:,})")

        # Verification
        verify_full_train_subsets(output_dir, args.seeds)

    else:
        # Default: create few-shot splits and test split
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
