"""
Tests for embedder configuration in tasks/jailbreak_detection/scripts/run_autointent.py.

Covers embedder naming, paths, metrics deduplication, and train() embedder wiring.
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def ra():
    """Import once (loads autointent and heavy deps)."""
    from tasks.jailbreak_detection.scripts import run_autointent

    return run_autointent


class TestGetModelName:
    def test_final_uses_e5large_suffix(self, ra):
        assert ra.get_model_name("classic-light", False) == "autointent_classic-light_e5large"

    def test_pilot_uses_pilot_suffix(self, ra):
        assert ra.get_model_name("classic-light", True) == "autointent_classic-light_pilot"

    def test_different_presets(self, ra):
        assert ra.get_model_name("nn-medium", False) == "autointent_nn-medium_e5large"
        assert ra.get_model_name("classic-medium", True) == "autointent_classic-medium_pilot"

    def test_with_query_prompt(self, ra):
        qp = "Classify if this is jailbreak: "
        name = ra.get_model_name("classic-light", False, qp)
        assert name.startswith("autointent_classic-light_e5large_qp_")
        assert "classify" in name.lower()

    def test_query_prompt_with_pilot(self, ra):
        qp = "Test prompt"
        name = ra.get_model_name("classic-light", True, qp)
        assert name.startswith("autointent_classic-light_pilot_qp_")


class TestGetEmbedderName:
    def test_pilot_uses_small_e5(self, ra):
        assert ra.get_embedder_name(True) == "intfloat/multilingual-e5-small"

    def test_final_uses_large_e5(self, ra):
        assert ra.get_embedder_name(False) == "intfloat/multilingual-e5-large-instruct"


class TestGetModelDir:
    def test_pilot_dir_differs_from_final(self, ra, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "get_runs_dir", lambda: tmp_path)
        final = ra.get_model_dir("classic-light", False, "fewshot", 10, 42)
        pilot = ra.get_model_dir("classic-light", True, "fewshot", 10, 42)
        assert final != pilot
        assert final.name == "autointent_classic-light_e5large_10shot_seed42"
        assert pilot.name == "autointent_classic-light_pilot_10shot_seed42"

    def test_full_mode_dir(self, ra, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "get_runs_dir", lambda: tmp_path)
        full_dir = ra.get_model_dir("classic-light", False, "full", None, 42)
        assert full_dir.name == "autointent_classic-light_e5large_full_seed42"


class TestSaveMetrics:
    def test_different_model_names_kept_separate(self, ra, tmp_path):
        """Distinct model_name keeps two metrics rows (pilot vs final)."""
        base = {
            "mode": "10shot",
            "n_shots": 10,
            "seed": 42,
            "f1": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "over_refusal_rate": 0.0,
        }
        ra.save_metrics({**base, "model_name": "autointent_classic-light_e5large"}, tmp_path)
        ra.save_metrics(
            {**base, "model_name": "autointent_classic-light_pilot"},
            tmp_path,
        )
        data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert len(data) == 2
        assert {r["model_name"] for r in data} == {
            "autointent_classic-light_e5large",
            "autointent_classic-light_pilot",
        }

    def test_run_metrics_filename_matches_metrics_row(self, ra, tmp_path):
        row = {
            "model_name": "autointent_classic-light_e5large",
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
            "metrics_autointent_classic-light_e5large_10shot_seed42.json"
        )
        assert json.loads(path.read_text(encoding="utf-8")) == row


class TestTrainEmbedderConfig:
    """Ensure EmbedderConfig is always applied (embedder is always fixed)."""

    @staticmethod
    def _minimal_train_dict():
        return {
            "intents": [{"id": 0, "name": "safe", "utterances": ["s"] * 10}],
            "oos_utterances": ["j"] * 10,
        }

    def test_embedder_config_always_set(self, ra, tmp_path, monkeypatch):
        """EmbedderConfig is always set (embedder is always fixed)."""
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
                no_automl_progress=True,
            )
            out = tmp_path / "run"
            out.mkdir(parents=True)
            ra.train(args, tmp_path / "data", out)

        assert "EmbedderConfig" in config_types
        assert "DataConfig" in config_types
        assert "LoggingConfig" in config_types

    def test_pilot_embedder_config_set(self, ra, tmp_path, monkeypatch):
        """EmbedderConfig is set with pilot embedder."""
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
                pilot=True,
                no_automl_progress=True,
            )
            out = tmp_path / "run"
            out.mkdir(parents=True)
            ra.train(args, tmp_path / "data", out)

        assert "EmbedderConfig" in config_types


def test_train_metadata_embedder_fixed_true(ra, tmp_path, monkeypatch):
    """Metadata always has embedder_fixed=True."""
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
            pilot=False,
            no_automl_progress=True,
        )
        out = tmp_path / "run"
        out.mkdir(parents=True)
        ra.train(args, tmp_path / "data", out)

    meta = json.loads((out / "train_metadata.json").read_text(encoding="utf-8"))
    assert meta["embedder_fixed"] is True
    assert meta["model_name"] == "autointent_classic-light_e5large"
    assert meta["embedder"] == "intfloat/multilingual-e5-large-instruct"


def test_train_metadata_pilot_embedder(ra, tmp_path, monkeypatch):
    """Pilot mode uses small embedder."""
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
            no_automl_progress=True,
        )
        out = tmp_path / "run"
        out.mkdir(parents=True)
        ra.train(args, tmp_path / "data", out)

    meta = json.loads((out / "train_metadata.json").read_text(encoding="utf-8"))
    assert meta["embedder_fixed"] is True
    assert meta["model_name"] == "autointent_classic-light_pilot"
    assert meta["embedder"] == "intfloat/multilingual-e5-small"
