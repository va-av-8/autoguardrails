"""
Build metrics_summary.csv from metrics.json and eval_scores files.

Usage:
    python scripts/build_metrics_summary.py
    python scripts/build_metrics_summary.py --output results/metrics_summary.csv

Reads:
    - results/metrics.json — main metrics storage
    - results/metrics_jailbreak_successful_14_05.json — additional metrics (different format)
    - runs/eval_scores_*.jsonl — per-example scores for ROC AUC computation

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
from sklearn.metrics import roc_auc_score, average_precision_score


def get_project_root() -> Path:
    """Get project root (jailbreak_detection task directory)."""
    return Path(__file__).parent.parent


def load_reference_y_true(runs_dir: Path) -> np.ndarray | None:
    """Load y_true from any eval_scores file (same eval set for all models)."""
    # Find any eval_scores file
    candidates = list(runs_dir.glob("eval_scores_*.jsonl"))
    # Prefer files without _qp suffix
    candidates = [c for c in candidates if "_qp" not in c.name]

    if not candidates:
        return None

    y_true_list = []
    with open(candidates[0], "r") as f:
        for line in f:
            row = json.loads(line)
            y_true_list.append(row["y_true"])

    return np.array(y_true_list)


def find_eval_scores_file(runs_dir: Path, model_name: str, mode: str, seed: int, n_shots: int | None) -> Path | None:
    """Find eval_scores file matching the run parameters."""
    # Build expected filename pattern
    model_name_safe = model_name.replace("/", "_").replace(" ", "_")

    if mode == "full":
        pattern = f"eval_scores_{model_name_safe}_full_seed{seed}.jsonl"
    else:
        pattern = f"eval_scores_{model_name_safe}_{n_shots}shot_seed{seed}.jsonl"

    path = runs_dir / pattern
    if path.exists():
        return path

    # Try glob for partial matches
    if mode == "full":
        candidates = list(runs_dir.glob(f"eval_scores_*{model_name_safe}*_full_seed{seed}.jsonl"))
    else:
        candidates = list(runs_dir.glob(f"eval_scores_*{model_name_safe}*_{n_shots}shot_seed{seed}.jsonl"))

    # Filter out files with _qp suffix (invalid prefix runs)
    candidates = [c for c in candidates if "_qp" not in c.name]

    if candidates:
        return candidates[0]

    return None


def compute_roc_auc_from_scores(eval_scores_path: Path) -> dict[str, float]:
    """Compute ROC AUC and PR AUC from eval_scores file."""
    rows = []
    with open(eval_scores_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    y_true = np.array([r["y_true"] for r in rows])

    # Get scores - handle different formats
    if "score_jb" in rows[0]:
        score_jb = np.array([r["score_jb"] for r in rows])
    elif "scores" in rows[0]:
        # Some formats have scores as a list [p_safe, p_jb]
        score_jb = np.array([r["scores"][1] if isinstance(r["scores"], list) else r["scores"] for r in rows])
    else:
        return {"roc_auc": None, "pr_auc": None}

    # Compute metrics
    try:
        roc_auc = roc_auc_score(y_true, score_jb)
    except ValueError:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_true, score_jb)
    except ValueError:
        pr_auc = None

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def parse_entry_standard(entry: dict) -> dict:
    """Parse entry in standard metrics.json format."""
    model_name = entry.get("model_name", "")
    mode = entry.get("mode", "")
    seed = entry.get("seed")
    n_shots = entry.get("n_shots")

    row = {
        "model_name": model_name,
        "mode": mode,
        "n_shots": n_shots,
        "seed": seed,
        "f1": entry.get("f1"),
        "precision": entry.get("precision"),
        "recall": entry.get("recall"),
        "over_refusal_rate": entry.get("over_refusal_rate"),
        "recall_adversarial_harmful": entry.get("recall_adversarial_harmful"),
        "timestamp": entry.get("timestamp"),
    }

    # Extra metrics
    extra = entry.get("extra", {})
    row["preset"] = extra.get("preset")
    row["embedder"] = extra.get("embedder")
    row["embedder_hf_model"] = extra.get("embedder_hf_model")
    row["embedder_fixed"] = extra.get("embedder_fixed")
    row["pilot"] = extra.get("pilot")

    # Eval counts
    eval_counts = extra.get("eval_counts", {})
    row["tp"] = eval_counts.get("tp") or eval_counts.get("TP")
    row["fp"] = eval_counts.get("fp") or eval_counts.get("FP")
    row["fn"] = eval_counts.get("fn") or eval_counts.get("FN")
    row["tn"] = eval_counts.get("tn") or eval_counts.get("TN")

    # Scores for ROC AUC computation
    row["_scores"] = extra.get("scores")
    row["_scores_eval_summary"] = extra.get("scores_eval_summary", {})

    return row


def parse_entry_flat(entry: dict) -> dict:
    """Parse entry in flat format (metrics_jailbreak_successful_14_05.json)."""
    row = {
        "model_name": entry.get("model_name", ""),
        "mode": entry.get("mode", ""),
        "n_shots": entry.get("n_shots"),
        "seed": entry.get("seed"),
        "f1": entry.get("f1"),
        "precision": entry.get("precision"),
        "recall": entry.get("recall"),
        "over_refusal_rate": entry.get("over_refusal_rate"),
        "recall_adversarial_harmful": entry.get("recall_adversarial_harmful"),
        "timestamp": entry.get("timestamp"),  # Usually None in this format
        "preset": entry.get("preset"),
        "embedder": entry.get("embedder"),
        "embedder_hf_model": entry.get("embedder_hf_model"),
        "embedder_fixed": entry.get("embedder_fixed"),
        "pilot": entry.get("pilot"),
        "tp": entry.get("tp"),
        "fp": entry.get("fp"),
        "fn": entry.get("fn"),
        "tn": entry.get("tn"),
        "_scores": None,  # Not available in flat format
        "_scores_eval_summary": {},
    }
    return row


def detect_format(data: list[dict]) -> str:
    """Detect format of metrics JSON: 'standard' or 'flat'."""
    if not data:
        return "unknown"
    first = data[0]
    if "extra" in first:
        return "standard"
    if "tp" in first and "extra" not in first:
        return "flat"
    return "unknown"


def build_summary(metrics_paths: list[Path], runs_dir: Path) -> pd.DataFrame:
    """Build summary DataFrame from metrics JSON files and eval_scores files."""
    # Load reference y_true for computing ROC AUC from extra.scores
    reference_y_true = load_reference_y_true(runs_dir)
    if reference_y_true is not None:
        print(f"Loaded reference y_true: {len(reference_y_true)} examples")

    rows = []
    for metrics_path in metrics_paths:
        if not metrics_path.exists():
            print(f"Warning: {metrics_path} not found, skipping")
            continue

        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        fmt = detect_format(metrics_data)
        print(f"Reading {metrics_path.name} ({len(metrics_data)} entries, format: {fmt})")

        for entry in metrics_data:
            # Parse entry based on format
            if fmt == "flat":
                row = parse_entry_flat(entry)
            else:
                row = parse_entry_standard(entry)

            model_name = row["model_name"]
            mode = row["mode"]
            seed = row["seed"]
            n_shots = row["n_shots"]

            # Extract internal fields and remove them from row
            scores = row.pop("_scores", None)
            scores_summary = row.pop("_scores_eval_summary", {})

            # Try to get ROC AUC from scores_eval_summary first
            row["roc_auc"] = scores_summary.get("roc_auc")
            row["pr_auc"] = None
            row["roc_auc_source"] = None

            # If no ROC AUC, try to compute from eval_scores file
            if row["roc_auc"] is None:
                eval_scores_path = find_eval_scores_file(runs_dir, model_name, mode, seed, n_shots)
                if eval_scores_path:
                    auc_metrics = compute_roc_auc_from_scores(eval_scores_path)
                    row["roc_auc"] = auc_metrics["roc_auc"]
                    row["pr_auc"] = auc_metrics["pr_auc"]
                    row["roc_auc_source"] = eval_scores_path.name

            # If still no ROC AUC, try to compute from scores in entry
            if row["roc_auc"] is None and scores is not None and reference_y_true is not None:
                if isinstance(scores, list) and len(scores) == len(reference_y_true):
                    scores_array = np.array(scores)
                    try:
                        row["roc_auc"] = roc_auc_score(reference_y_true, scores_array)
                        row["pr_auc"] = average_precision_score(reference_y_true, scores_array)
                        row["roc_auc_source"] = "extra.scores"
                    except ValueError:
                        pass

            if row["roc_auc"] is not None and row["roc_auc_source"] is None:
                row["roc_auc_source"] = "metrics_json"

            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by model_name, mode, n_shots, seed
    df = df.sort_values(["model_name", "mode", "n_shots", "seed"]).reset_index(drop=True)

    return df


def build_aggregated_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build aggregated summary with mean and std across seeds for each model+mode."""
    # Metrics to aggregate
    metrics = ["f1", "precision", "recall", "over_refusal_rate", "roc_auc", "pr_auc"]

    # Group by model_name and mode
    grouped = df.groupby(["model_name", "mode"])

    rows = []
    for (model_name, mode), group in grouped:
        row = {
            "model_name": model_name,
            "mode": mode,
            "n_seeds": len(group),
            "seeds": sorted(group["seed"].dropna().astype(int).tolist()),
        }

        # Compute mean and std for each metric
        for metric in metrics:
            values = group[metric].dropna()
            if len(values) > 0:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std() if len(values) > 1 else 0.0
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None

        rows.append(row)

    agg_df = pd.DataFrame(rows)

    # Sort by model_name, mode
    mode_order = {"10shot": 0, "20shot": 1, "50shot": 2, "full": 3}
    agg_df["mode_order"] = agg_df["mode"].map(lambda x: mode_order.get(x, 99))
    agg_df = agg_df.sort_values(["model_name", "mode_order"]).drop(columns=["mode_order"]).reset_index(drop=True)

    return agg_df


def format_mean_std(mean: float | None, std: float | None, decimals: int = 3) -> str:
    """Format mean ± std as string."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    if std is None or std == 0 or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description="Build metrics summary CSV from metrics.json and eval_scores files")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        nargs="+",
        default=None,
        help="Path(s) to metrics JSON files (default: results/metrics.json + results/metrics_jailbreak_successful_14_05.json)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Path to runs directory (default: runs/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: results/metrics_summary.csv)",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    runs_dir = args.runs_dir or (project_root / "runs")
    output_path = args.output or (project_root / "results" / "metrics_summary.csv")

    # Default metrics files
    if args.metrics_json:
        metrics_paths = args.metrics_json
    else:
        metrics_paths = [
            project_root / "results" / "metrics.json",
            project_root / "results" / "metrics_jailbreak_successful_14_05.json",
        ]

    print(f"Reading eval_scores from: {runs_dir}")

    df = build_summary(metrics_paths, runs_dir)

    # Summary stats
    print(f"\nTotal entries: {len(df)}")
    print(f"With ROC AUC: {df['roc_auc'].notna().sum()}")
    print(f"Without ROC AUC: {df['roc_auc'].isna().sum()}")

    # Show modes
    print(f"\nBy mode:")
    print(df.groupby("mode").size().to_string())

    # Show models
    print(f"\nBy model:")
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

    # Show sample of per-run metrics
    print(f"\nPer-run metrics (first 5 rows):")
    display_cols = ["model_name", "mode", "seed", "f1", "roc_auc", "over_refusal_rate"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].head().to_string(index=False))

    # Show aggregated summary
    print(f"\n{'='*80}")
    print("AGGREGATED METRICS (mean ± std across seeds)")
    print(f"{'='*80}\n")

    # Display formatted aggregated results
    for model_name in agg_df["model_name"].unique():
        print(f"\n{model_name}")
        print("-" * len(model_name))
        model_data = agg_df[agg_df["model_name"] == model_name]

        for _, row in model_data.iterrows():
            mode = row["mode"]
            n_seeds = row["n_seeds"]
            f1 = format_mean_std(row["f1_mean"], row["f1_std"])
            roc_auc = format_mean_std(row["roc_auc_mean"], row["roc_auc_std"])
            precision = format_mean_std(row["precision_mean"], row["precision_std"])
            recall = format_mean_std(row["recall_mean"], row["recall_std"])
            over_refusal = format_mean_std(row["over_refusal_rate_mean"], row["over_refusal_rate_std"])

            print(f"  {mode:8s} (n={n_seeds}): F1={f1:18s} ROC-AUC={roc_auc:18s} P={precision:18s} R={recall:18s} ORR={over_refusal}")

    # Also show as table
    print(f"\n{'='*80}")
    print("AGGREGATED TABLE")
    print(f"{'='*80}\n")
    display_cols = ["model_name", "mode", "n_seeds", "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std", "over_refusal_rate_mean", "over_refusal_rate_std"]
    print(agg_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
