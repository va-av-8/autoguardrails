"""
Build metrics_summary.csv from metrics.json.

Usage:
    python scripts/build_metrics_summary.py
    python scripts/build_metrics_summary.py --output results/metrics_summary.csv

Reads:
    - results/metrics.json — main metrics storage

Outputs:
    - results/metrics_summary.csv — consolidated metrics table (per-run)
    - results/metrics_summary_agg.csv — aggregated metrics (mean ± std across seeds)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def get_task_root() -> Path:
    """Get task root (oos_detection directory)."""
    return Path(__file__).parent.parent


# Column order: logical grouping
COLUMN_ORDER = [
    # Identification
    "model_name",
    "source",
    "mode",
    "n_shots",
    "seed",
    "prediction_mode",
    "preset",
    # Main metrics
    "oos_recall",
    "in_domain_acc",
    "f1_oos",
    "accuracy",
    "macro_f1",
    "oos_precision",
    # Secondary metrics
    "auroc",
    "au_ioc",
    # Timing
    "latency_ms",
    "train_sec",
    "fit_sec",
    "calibrate_sec",
    "eval_sec",
    # Meta
    "framework",
    "embedder",
    "embedder_key",
    "embedder_fixed",
    "decision_metric",
    "pilot",
    "model_dir",
    "hypothesis",
    "kaggle_run",
    "comparable_to_table3",
    "is_reference",
    "timestamp",
]

# Metrics to aggregate (mean ± std)
METRICS_TO_AGGREGATE = [
    "oos_recall",
    "in_domain_acc",
    "f1_oos",
    "accuracy",
    "macro_f1",
    "oos_precision",
    "auroc",
    "au_ioc",
    "latency_ms",
    "train_sec",
    "fit_sec",
    "calibrate_sec",
    "eval_sec",
]


def parse_entry(entry: dict) -> dict:
    """Parse entry from metrics.json into flat row with all fields."""
    extra = entry.get("extra", {})

    row = {}

    # Root-level fields
    for key in ["model_name", "mode", "n_shots", "seed", "oos_recall", "in_domain_acc",
                "f1_oos", "auroc", "au_ioc", "latency_ms", "is_reference", "timestamp"]:
        row[key] = entry.get(key)

    # Extra fields (flattened)
    for key in ["source", "prediction_mode", "preset", "accuracy", "macro_f1",
                "oos_precision", "train_sec", "fit_sec", "calibrate_sec", "eval_sec",
                "framework", "embedder", "embedder_key", "embedder_fixed",
                "decision_metric", "pilot", "model_dir", "hypothesis", "kaggle_run",
                "comparable_to_table3"]:
        row[key] = extra.get(key)

    return row


def build_summary(metrics_path: Path) -> pd.DataFrame:
    """Build summary DataFrame from metrics.json."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    print(f"Reading {metrics_path.name} ({len(metrics_data)} entries)")

    rows = [parse_entry(entry) for entry in metrics_data]
    df = pd.DataFrame(rows)

    # Reorder columns: COLUMN_ORDER first, then any extra columns alphabetically
    existing_cols = set(df.columns)
    ordered_cols = [c for c in COLUMN_ORDER if c in existing_cols]
    extra_cols = sorted(existing_cols - set(ordered_cols))
    df = df[ordered_cols + extra_cols]

    # Sort by model_name, source, mode, n_shots, seed
    mode_order = {"full": 0, "10shot": 1, "20shot": 2, "50shot": 3}
    df["_mode_order"] = df["mode"].map(lambda x: mode_order.get(x, 99))
    df = df.sort_values(
        ["model_name", "source", "_mode_order", "n_shots", "seed"]
    ).drop(columns=["_mode_order"]).reset_index(drop=True)

    return df


def build_aggregated_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build aggregated summary with mean and std across seeds for each model+source+mode."""
    grouped = df.groupby(["model_name", "source", "mode"], dropna=False)

    rows = []
    for (model_name, source, mode), group in grouped:
        row = {
            "model_name": model_name,
            "source": source,
            "mode": mode,
            "n_seeds": len(group),
            "seeds": sorted([int(s) for s in group["seed"].dropna().tolist()]) or None,
        }

        for metric in METRICS_TO_AGGREGATE:
            if metric not in group.columns:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                continue
            values = group[metric].dropna()
            if len(values) > 0:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std() if len(values) > 1 else 0.0
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None

        rows.append(row)

    agg_df = pd.DataFrame(rows)

    # Sort
    mode_order = {"full": 0, "10shot": 1, "20shot": 2, "50shot": 3}
    agg_df["_mode_order"] = agg_df["mode"].map(lambda x: mode_order.get(x, 99))
    agg_df = agg_df.sort_values(
        ["model_name", "source", "_mode_order"]
    ).drop(columns=["_mode_order"]).reset_index(drop=True)

    return agg_df


def format_mean_std(mean: float | None, std: float | None, decimals: int = 4) -> str:
    """Format mean ± std as string."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    if std is None or std == 0 or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def print_aggregated_report(agg_df: pd.DataFrame) -> None:
    """Print formatted aggregated report to stdout."""
    print(f"\n{'=' * 100}")
    print("AGGREGATED METRICS (mean ± std across seeds)")
    print(f"{'=' * 100}")

    for model_name in agg_df["model_name"].unique():
        model_data = agg_df[agg_df["model_name"] == model_name]

        print(f"\n{model_name}")
        print("-" * len(model_name))

        for _, row in model_data.iterrows():
            source = row["source"] or "-"
            mode = row["mode"]
            n_seeds = row["n_seeds"]

            f1 = format_mean_std(row["f1_oos_mean"], row["f1_oos_std"])
            recall = format_mean_std(row["oos_recall_mean"], row["oos_recall_std"])
            auroc = format_mean_std(row["auroc_mean"], row["auroc_std"])

            print(
                f"  {source:15s} {mode:8s} (n={n_seeds}): "
                f"F1={f1:20s} Recall={recall:20s} AUROC={auroc:20s}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build metrics summary CSV from metrics.json"
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Path to metrics.json (default: results/metrics.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: results/metrics_summary.csv)",
    )
    args = parser.parse_args()

    task_root = get_task_root()
    metrics_path = args.metrics_json or (task_root / "results" / "metrics.json")
    output_path = args.output or (task_root / "results" / "metrics_summary.csv")

    # Build per-run summary
    df = build_summary(metrics_path)

    # Stats
    print(f"\nTotal entries: {len(df)}")
    print(f"\nBy source:")
    print(df.groupby("source", dropna=False).size().to_string())
    print(f"\nBy mode:")
    print(df.groupby("mode").size().to_string())
    print(f"\nBy model_name:")
    print(df.groupby("model_name").size().to_string())

    # Save per-run summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Build and save aggregated summary
    agg_df = build_aggregated_summary(df)
    agg_output_path = output_path.parent / "metrics_summary_agg.csv"
    agg_df.to_csv(agg_output_path, index=False)
    print(f"Saved: {agg_output_path}")

    # Print aggregated report
    print_aggregated_report(agg_df)

    # Show sample
    print(f"\n{'=' * 100}")
    print("PER-RUN METRICS (first 10 rows)")
    print(f"{'=' * 100}\n")
    display_cols = ["model_name", "source", "mode", "seed", "f1_oos", "oos_recall", "auroc"]
    print(df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
