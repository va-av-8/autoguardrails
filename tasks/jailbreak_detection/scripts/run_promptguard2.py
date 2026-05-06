"""
Evaluation of PromptGuard 2 on WildJailbreak test set.

Model: meta-llama/Llama-Prompt-Guard-2-86M
Type: BERT-style classifier (text-classification pipeline)
Source: LlamaFirewall (arxiv:2505.03574)

Usage:
    python scripts/run_promptguard2.py

    With custom paths:
    python scripts/run_promptguard2.py \
        --eval_binary data/processed/wildjailbreak_eval_binary.jsonl \
        --output_dir results/ \
        --batch_size 32 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent
task_dir = script_dir.parent
project_root = task_dir.parent.parent
sys.path.insert(0, str(project_root))

from tasks.jailbreak_detection.src.metrics import evaluate_jailbreak


def load_eval_binary(path: Path) -> list[dict]:
    """Load wildjailbreak_eval_binary.jsonl."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def chunk_text(text: str, tokenizer, max_length: int = 512, overlap: int = 50) -> list[str]:
    """
    Split text into chunks that fit within max_length tokens.
    Uses overlap to avoid cutting important context.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_length - 2:  # Account for [CLS] and [SEP]
        return [text]

    chunks = []
    stride = max_length - 2 - overlap

    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + max_length - 2]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        if i + max_length - 2 >= len(tokens):
            break

    return chunks


def predict_with_chunking(
    texts: list[str],
    pipeline,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    overlap: int = 50,
    label_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run PromptGuard 2 on texts with chunking for long prompts.
    Returns (predictions, scores) where score is max jailbreak probability.
    """
    if label_map is None:
        label_map = {"BENIGN": 0, "JAILBREAK": 1}

    jailbreak_label = [k for k, v in label_map.items() if v == 1][0]

    predictions = []
    scores = []

    for text in tqdm(texts, desc="PromptGuard 2"):
        chunks = chunk_text(text, tokenizer, max_length, overlap)

        # Process chunks in batches
        chunk_scores = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            results = pipeline(batch)

            for result in results:
                # Get jailbreak score
                if isinstance(result, list):
                    result = result[0]

                if result["label"] == jailbreak_label:
                    chunk_scores.append(result["score"])
                else:
                    chunk_scores.append(1 - result["score"])

        # Max score across chunks (most jailbreak-like segment)
        max_score = max(chunk_scores)
        scores.append(max_score)
        predictions.append(1 if max_score > 0.5 else 0)

    return np.array(predictions), np.array(scores)


def save_results(results: dict, output_dir: Path) -> None:
    """Save results to metrics.json, appending to existing list."""
    output_path = output_dir / "metrics.json"

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        if not isinstance(all_results, list):
            # Convert dict format to list if needed
            all_results = list(all_results.values()) if all_results else []
    else:
        all_results = []

    # Remove existing entry with same model_name
    all_results = [r for r in all_results if r.get("model_name") != results["model_name"]]
    all_results.append(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PromptGuard 2 on WildJailbreak"
    )
    parser.add_argument(
        "--eval_binary",
        type=Path,
        default=None,
        help="Path to wildjailbreak_eval_binary.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Path to results directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device for inference",
    )
    args = parser.parse_args()

    # Set default paths
    if args.eval_binary is None:
        args.eval_binary = task_dir / "data" / "processed" / "wildjailbreak_eval_binary.jsonl"
    if args.output_dir is None:
        args.output_dir = task_dir / "results"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.eval_binary}")
    eval_data = load_eval_binary(args.eval_binary)

    prompts = [r["prompt"] for r in eval_data]
    y_true = np.array([1 if r["binary_label"] == "jailbreak" else 0 for r in eval_data])
    data_types = np.array([r["data_type"] for r in eval_data])

    print(f"Loaded {len(prompts)} examples")
    print(f"  safe: {sum(y_true == 0)}, jailbreak: {sum(y_true == 1)}")

    # Load model
    print("\nLoading PromptGuard 2...")
    from transformers import pipeline as hf_pipeline, AutoTokenizer

    model_name = "meta-llama/Llama-Prompt-Guard-2-86M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = hf_pipeline(
        "text-classification",
        model=model_name,
        tokenizer=tokenizer,
        device=0 if args.device == "cuda" else -1 if args.device == "cpu" else args.device,
    )

    # Show raw predictions on first 3 examples
    print("\n=== Raw predictions on first 3 examples ===")
    for i in range(min(3, len(prompts))):
        result = classifier(prompts[i][:500])  # Truncate for display
        print(f"Example {i+1}:")
        print(f"  True label: {['safe', 'jailbreak'][y_true[i]]}")
        print(f"  Model output: {result}")
        print()

    # Determine label mapping from model output
    sample_result = classifier(prompts[0])
    if isinstance(sample_result, list):
        sample_result = sample_result[0]

    available_labels = [sample_result["label"]]
    # Try to get all labels from model config
    if hasattr(classifier.model.config, "id2label"):
        available_labels = list(classifier.model.config.id2label.values())

    print(f"Available labels: {available_labels}")

    # Build label map (adjust based on actual model output)
    label_map = {}
    for label in available_labels:
        label_upper = label.upper()
        if "BENIGN" in label_upper or "SAFE" in label_upper:
            label_map[label] = 0
        elif "JAILBREAK" in label_upper or "MALICIOUS" in label_upper or "INJECTION" in label_upper:
            label_map[label] = 1

    if len(label_map) < 2:
        # Fallback: assume binary with scores
        print("Warning: Could not determine label mapping, using score threshold")
        label_map = {available_labels[0]: 1}  # Assume positive class

    print(f"Label mapping: {label_map}")

    # Run inference
    print(f"\nRunning inference on {len(prompts)} examples...")
    y_pred, scores = predict_with_chunking(
        prompts,
        classifier,
        tokenizer,
        batch_size=args.batch_size,
        max_length=512,
        overlap=50,
        label_map=label_map,
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics = evaluate_jailbreak(y_true, y_pred, data_types, oos_label=1)

    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    results = {
        "model_name": "promptguard2_86m",
        "model": model_name,
        "mode": "zero-shot",
        "test_size": len(prompts),
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "over_refusal_rate": metrics["over_refusal_rate"],
        "recall_adversarial_harmful": metrics.get("recall_adversarial_harmful"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
