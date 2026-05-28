"""Tests for full train mode and --all-seeds in run_autointent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def ra():
    from tasks.jailbreak_detection.scripts import run_autointent

    return run_autointent


class TestFullDataPaths:
    def test_full_train_filename(self, ra):
        assert ra.full_train_filename(42) == "wildjailbreak_full100k_seed42.json"

    def test_default_seeds_order(self, ra):
        assert ra.DEFAULT_SEEDS == (42, 123, 456)

    def test_get_model_name_classic_medium(self, ra):
        assert ra.get_model_name("classic-medium", False, False) == "autointent_classic-medium"
        assert ra.get_model_name("nn-medium", False, True) == "autointent_nn-medium_autoembedder"

    def test_model_dir_full(self, ra, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "get_runs_dir", lambda: tmp_path)
        p = ra.get_model_dir("classic-light", False, False, "full", None, 123)
        assert p.name == "autointent_classic-light_full_seed123"


class TestMainAllSeeds:
    def test_train_only_loops_all_seeds(self, ra, monkeypatch, tmp_path):
        seeds_seen: list[int] = []

        def fake_train(a, d, m):
            seeds_seen.append(a.seed)

        monkeypatch.setattr(ra, "get_data_dir", lambda: tmp_path / "data")
        monkeypatch.setattr(ra, "get_results_dir", lambda: tmp_path / "res")
        monkeypatch.setattr(ra, "train", fake_train)
        old = sys.argv
        try:
            sys.argv = ["run_autointent.py", "--mode", "fewshot", "--all-seeds", "--train-only"]
            ra.main()
        finally:
            sys.argv = old

        assert seeds_seen == list(ra.DEFAULT_SEEDS)


class TestEvaluateFullMetricsExtra:
    """metrics.json rows for mode=full include extra (embedder, preset, model_dir)."""

    def test_full_record_has_extra_with_embedder(self, ra, tmp_path, monkeypatch):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "train_metadata.json").write_text(
            json.dumps(
                {
                    "model_name": "autointent_classic-light",
                    "mode": "full",
                    "n_shots": None,
                    "seed": 42,
                    "embedder": "intfloat/multilingual-e5-large-instruct",
                    "embedder_fixed": True,
                    "pilot": False,
                    "preset": "classic-light",
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            ra,
            "load_test",
            lambda d: {"utterances": ["hello"], "labels": [0]},
        )
        monkeypatch.setattr(
            ra,
            "load_eval_binary",
            lambda d: [{"binary_label": "safe", "data_type": "vanilla_benign"}],
        )

        mock_pipe = MagicMock()
        mock_pipe.predict.return_value = [0]

        monkeypatch.setattr(
            ra,
            "evaluate_jailbreak",
            lambda *a, **k: {
                "f1": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "over_refusal_rate": 0.0,
                "recall_adversarial_harmful": None,
            },
        )

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        args = argparse.Namespace(
            preset="classic-light",
            mode="full",
            pilot=False,
            no_fix_embedder=False,
            seed=42,
        )

        with patch.object(ra, "Pipeline") as mock_pipeline:
            mock_pipeline.load.return_value = mock_pipe
            ra.evaluate(args, tmp_path / "data", model_dir, results_dir, runs_dir=runs_dir)

        rows = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
        assert len(rows) >= 1
        last = rows[-1]
        assert last["mode"] == "full"
        assert last["n_shots"] is None
        assert "extra" in last
        assert last["extra"]["embedder"] == "intfloat/multilingual-e5-large-instruct"
        assert last["extra"]["preset"] == "classic-light"
        assert last["extra"]["embedder_fixed"] is True
        assert last["extra"]["pilot"] is False
        assert Path(last["extra"]["model_dir"]) == model_dir

        per_run = runs_dir / ra.run_metrics_filename(last)
        assert per_run.exists()
        disk = json.loads(per_run.read_text(encoding="utf-8"))
        assert disk == last


def test_help_lists_full_and_all_seeds(ra):
    import sys
    from io import StringIO

    old_argv = sys.argv
    old_stdout = sys.stdout
    buf = StringIO()
    try:
        sys.argv = ["run_autointent.py", "--help"]
        sys.stdout = buf
        with pytest.raises(SystemExit) as exc:
            ra.main()
        assert exc.value.code == 0
    finally:
        sys.stdout = old_stdout
        sys.argv = old_argv

    out = buf.getvalue()
    assert "full" in out
    assert "--all-seeds" in out
    assert "--print-metrics-json" in out
    assert "--preset" in out


class TestSaveMetricsDedupFull:
    def test_full_and_fewshot_same_seed_distinct_rows(self, ra, tmp_path):
        base = {
            "seed": 42,
            "f1": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "over_refusal_rate": 0.0,
            "timestamp": "t",
        }
        ra.save_metrics({**base, "model_name": "m", "mode": "full", "n_shots": None}, tmp_path)
        ra.save_metrics(
            {**base, "model_name": "m", "mode": "10shot", "n_shots": 10},
            tmp_path,
        )
        data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert len(data) == 2
