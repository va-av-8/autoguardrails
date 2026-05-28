"""
Tests for embedder flags in tasks/jailbreak_detection/scripts/run_autointent.py.

Covers --no-fix-embedder naming, paths, metrics deduplication, and train() embedder wiring.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def ra():
    """Import once (loads autointent and heavy deps)."""
    from tasks.jailbreak_detection.scripts import run_autointent

    return run_autointent


class TestGetModelName:
    def test_final_fixed_embedder(self, ra):
        assert ra.get_model_name("classic-light", False, False) == "autointent_classic-light"

    def test_pilot_fixed_embedder(self, ra):
        assert ra.get_model_name("classic-light", True, False) == "autointent_classic-light_pilot"

    def test_autoembedder_overrides_pilot(self, ra):
        assert ra.get_model_name("classic-light", False, True) == "autointent_classic-light_autoembedder"
        assert ra.get_model_name("classic-light", True, True) == "autointent_classic-light_autoembedder"


class TestGetEmbedderName:
    def test_pilot_uses_small_e5(self, ra):
        assert ra.get_embedder_name(True) == "intfloat/multilingual-e5-small"

    def test_final_uses_large_e5(self, ra):
        assert ra.get_embedder_name(False) == "intfloat/multilingual-e5-large-instruct"


class TestGetModelDir:
    def test_autoembedder_dir_differs_from_fixed_same_shots_seed(
        self, ra, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ra, "get_runs_dir", lambda: tmp_path)
        fixed = ra.get_model_dir("classic-light", False, False, "fewshot", 10, 42)
        auto = ra.get_model_dir("classic-light", False, True, "fewshot", 10, 42)
        assert fixed != auto
        assert "autoembedder" not in fixed.name
        assert auto.name == "autointent_classic-light_autoembedder_10shot_seed42"
        assert fixed.name == "autointent_classic-light_10shot_seed42"


class TestSaveMetrics:
    def test_same_seed_two_embedder_modes_both_kept(self, ra, tmp_path):
        """Distinct model_name keeps two metrics rows (fixed vs autoembedder)."""
        base = {
            "mode": "10shot",
            "n_shots": 10,
            "seed": 42,
            "f1": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "over_refusal_rate": 0.0,
        }
        ra.save_metrics({**base, "model_name": "autointent_classic-light"}, tmp_path)
        ra.save_metrics(
            {**base, "model_name": "autointent_classic-light_autoembedder"},
            tmp_path,
        )
        data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert len(data) == 2
        assert {r["model_name"] for r in data} == {
            "autointent_classic-light",
            "autointent_classic-light_autoembedder",
        }

    def test_run_metrics_filename_matches_metrics_row(self, ra, tmp_path):
        row = {
            "model_name": "autointent_classic-light_autoembedder",
            "mode": "10shot",
            "n_shots": 10,
            "seed": 42,
            "f1": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "over_refusal_rate": 0.1,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "extra": {"model_dir": "/x"},
        }
        path = ra.save_run_metrics_file(row, tmp_path)
        assert path.name == (
            "metrics_autointent_classic-light_autoembedder_10shot_seed42.json"
        )
        assert json.loads(path.read_text(encoding="utf-8")) == row


class TestTrainEmbedderConfig:
    """Ensure EmbedderConfig is only applied when embedder is fixed."""

    @staticmethod
    def _minimal_train_dict():
        return {
            "intents": [{"id": 0, "name": "safe", "utterances": ["s"] * 10}],
            "oos_utterances": ["j"] * 10,
        }

    def test_no_fix_embedder_skips_embedder_config(self, ra, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ra,
            "load_fewshot_train",
            lambda n, s, d: TestTrainEmbedderConfig._minimal_train_dict(),
        )
        monkeypatch.setattr(
            ra,
            "load_test",
            lambda d: {"utterances": ["u1", "u2"], "labels": [0, 1]},
        )

        pipeline_inst = MagicMock()
        config_types: list[str] = []

        def track_config(cfg):
            config_types.append(type(cfg).__name__)

        pipeline_inst.set_config.side_effect = track_config

        with patch.object(ra, "Pipeline") as mock_pipe:
            mock_pipe.from_preset.return_value = pipeline_inst
            args = argparse.Namespace(
                preset="classic-light",
                mode="fewshot",
                n_shots=10,
                seed=42,
                pilot=False,
                no_fix_embedder=True,
                no_automl_progress=True,
            )
            out = tmp_path / "run"
            out.mkdir(parents=True)
            ra.train(args, tmp_path / "data", out)

        assert "EmbedderConfig" not in config_types
        assert "DataConfig" in config_types
        assert "LoggingConfig" in config_types

    def test_fixed_embedder_sets_embedder_config(self, ra, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ra,
            "load_fewshot_train",
            lambda n, s, d: TestTrainEmbedderConfig._minimal_train_dict(),
        )
        monkeypatch.setattr(
            ra,
            "load_test",
            lambda d: {"utterances": ["u1", "u2"], "labels": [0, 1]},
        )

        pipeline_inst = MagicMock()
        config_types: list[str] = []

        def track_config(cfg):
            config_types.append(type(cfg).__name__)

        pipeline_inst.set_config.side_effect = track_config

        with patch.object(ra, "Pipeline") as mock_pipe:
            mock_pipe.from_preset.return_value = pipeline_inst
            args = argparse.Namespace(
                preset="classic-light",
                mode="fewshot",
                n_shots=10,
                seed=42,
                pilot=False,
                no_fix_embedder=False,
                no_automl_progress=True,
            )
            out = tmp_path / "run"
            out.mkdir(parents=True)
            ra.train(args, tmp_path / "data", out)

        assert "EmbedderConfig" in config_types


def test_cli_help_lists_no_fix_embedder(ra):
    """argparse help includes --no-fix-embedder (no subprocess; uses autointent stub)."""
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
    assert "--no-fix-embedder" in out


def test_train_metadata_written_with_embedder_fixed_false(ra, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ra,
        "load_fewshot_train",
        lambda n, s, d: TestTrainEmbedderConfig._minimal_train_dict(),
    )
    monkeypatch.setattr(
        ra,
        "load_test",
        lambda d: {"utterances": ["u1", "u2"], "labels": [0, 1]},
    )

    pipeline_inst = MagicMock()
    pipeline_inst.set_config = MagicMock()

    with patch.object(ra, "Pipeline") as mock_pipe:
        mock_pipe.from_preset.return_value = pipeline_inst
        args = argparse.Namespace(
            preset="classic-light",
            mode="fewshot",
            n_shots=10,
            seed=42,
            pilot=True,
            no_fix_embedder=True,
            no_automl_progress=True,
        )
        out = tmp_path / "run"
        out.mkdir(parents=True)
        ra.train(args, tmp_path / "data", out)

    meta = json.loads((out / "train_metadata.json").read_text(encoding="utf-8"))
    assert meta["embedder_fixed"] is False
    assert meta["model_name"] == "autointent_classic-light_autoembedder"
    assert meta["embedder"] == "auto (optimized by AutoML)"
