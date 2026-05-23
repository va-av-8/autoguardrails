"""
Data loading utilities for Jailbreak Detection AutoML experiments.

Loads data in the format expected by BinaryFrameworkWrapper:
- train: (texts, labels) where labels are 0=safe, 1=jailbreak
- test: (texts, labels)
- data_types: list of data_type strings for subset metrics
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("tasks/jailbreak_detection/data/processed")
DEFAULT_SEEDS = [42, 123, 456]
EVAL_TYPES_FILE = DATA_DIR / "wildjailbreak_eval_binary.jsonl"


def _parse_n_shots_from_mode(mode: str) -> int | None:
    """Extract n_shots from mode string like '10shot', '20shot', '50shot'."""
    if mode.endswith("shot"):
        try:
            return int(mode[:-4])
        except ValueError:
            pass
    return None


def load_train(
    mode: str,
    seed: int,
    n_shots: int | None = None,
) -> tuple[list[str], list[int]]:
    """
    Load training data.

    Args:
        mode: One of "10shot", "20shot", "50shot", "full"
        seed: Random seed (42, 123, 456)
        n_shots: Explicit n_shots value. If None, parsed from mode.

    Returns:
        (texts, labels) where labels are 0=safe, 1=jailbreak

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If mode is invalid
    """
    if mode == "full":
        path = DATA_DIR / f"wildjailbreak_full100k_seed{seed}.json"
    else:
        # Parse n_shots from mode or use explicit value
        if n_shots is None:
            n_shots = _parse_n_shots_from_mode(mode)
        if n_shots is None:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected '10shot', '20shot', '50shot', or 'full'"
            )
        path = DATA_DIR / f"train_shot{n_shots}_seed{seed}.json"

    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert AutoIntent format to (texts, labels)
    # intents[0].utterances → label 0 (safe)
    # oos_utterances → label 1 (jailbreak)
    texts: list[str] = []
    labels: list[int] = []

    # Safe examples (label=0)
    for intent in data.get("intents", []):
        for utt in intent.get("utterances", []):
            texts.append(utt)
            labels.append(0)

    # Jailbreak examples (label=1)
    for utt in data.get("oos_utterances", []):
        texts.append(utt)
        labels.append(1)

    return texts, labels


def load_test() -> tuple[list[str], list[int]]:
    """
    Load test data.

    Returns:
        (texts, labels) where labels are 0=safe, 1=jailbreak

    Raises:
        FileNotFoundError: If test.json doesn't exist
    """
    path = DATA_DIR / "test.json"

    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data["utterances"]
    labels = data["labels"]

    return texts, labels


def load_eval_data_types() -> list[str]:
    """
    Load data_type values for each test example.

    Used for computing subset metrics (recall on adversarial_harmful, etc.).

    Returns:
        List of data_type strings, length should match test.json

    Raises:
        FileNotFoundError: If eval binary file doesn't exist
    """
    if not EVAL_TYPES_FILE.exists():
        raise FileNotFoundError(f"Eval types file not found: {EVAL_TYPES_FILE}")

    data_types: list[str] = []
    with open(EVAL_TYPES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            data_types.append(record["data_type"])

    return data_types


def compute_split_id(
    mode: str,
    seed: int,
    n_shots: int | None = None,
) -> str:
    """
    Compute split ID for embedding cache.

    Args:
        mode: One of "10shot", "20shot", "50shot", "full"
        seed: Random seed
        n_shots: Explicit n_shots value. If None, parsed from mode.

    Returns:
        Split ID string like "train_shot10_seed42" or "full100k_seed42"
    """
    if mode == "full":
        return f"full100k_seed{seed}"

    # Parse n_shots from mode or use explicit value
    if n_shots is None:
        n_shots = _parse_n_shots_from_mode(mode)
    if n_shots is None:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected '10shot', '20shot', '50shot', or 'full'"
        )

    return f"train_shot{n_shots}_seed{seed}"
