#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OOS metrics results.")
    parser.add_argument(
        "--results-file",
        default="tasks/oos_detection/results/metrics.json",
        help="Path to metrics.json file.",
    )
    parser.add_argument("--model-name", default=None, help="Filter by model_name.")
    parser.add_argument("--source", default=None, help="Filter by extra.source.")
    parser.add_argument("--mode", default=None, help="Filter by mode.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Print only top-k rows after sorting by f1_oos.",
    )
    return parser.parse_args()


def _extract_source(extra: object) -> str | None:
    if isinstance(extra, dict):
        value = extra.get("source")
        return str(value) if value is not None else None
    return None


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_file)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not data:
        print("No results found.")
        return

    df = pd.DataFrame(data)
    if "extra" in df.columns:
        df["source"] = df["extra"].apply(_extract_source)
    elif "source" not in df.columns:
        df["source"] = None

    if args.model_name:
        df = df[df["model_name"] == args.model_name]
    if args.source:
        df = df[df["source"] == args.source]
    if args.mode:
        df = df[df["mode"] == args.mode]

    if df.empty:
        print("No rows after filtering.")
        return

    df = df.sort_values(by="f1_oos", ascending=False)
    if args.top_k is not None:
        df = df.head(args.top_k)

    columns = [
        "model_name",
        "source",
        "mode",
        "n_shots",
        "seed",
        "f1_oos",
        "oos_recall",
        "in_domain_acc",
        "auroc",
        "au_ioc",
        "latency_ms",
    ]
    existing = [col for col in columns if col in df.columns]
    print(df[existing].to_string(index=False))


if __name__ == "__main__":
    main()
