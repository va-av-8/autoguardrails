"""Tests for preset aliases and zero-shot intent descriptions."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def ra():
    from tasks.jailbreak_detection.scripts import run_autointent

    return run_autointent


class TestResolvePreset:
    def test_bert_finetune_alias(self, ra):
        cli, internal = ra.resolve_preset("bert-finetune")
        assert cli == "bert-finetune"
        assert internal == "transformers-light"

    def test_zero_shot_unchanged(self, ra):
        cli, internal = ra.resolve_preset("zero-shot-encoders")
        assert cli == internal == "zero-shot-encoders"

    def test_unknown_raises(self, ra):
        with pytest.raises(ValueError, match="Unknown preset"):
            ra.resolve_preset("not-a-preset")


class TestBuildBinaryIntents:
    def test_without_descriptions(self, ra):
        intents = ra.build_binary_intents(with_descriptions=False)
        assert len(intents) == 2
        assert intents[0]["name"] == "safe"
        assert intents[1]["name"] == "jailbreak"
        assert "description" not in intents[0]
        assert "description" not in intents[1]

    def test_with_descriptions_for_zero_shot(self, ra):
        intents = ra.build_binary_intents(with_descriptions=True)
        assert all("description" in i and i["description"] for i in intents)
        assert ra.preset_needs_intent_descriptions("zero-shot-encoders")
        assert not ra.preset_needs_intent_descriptions("classic-medium")


class TestMetricsExport:
    def test_metrics_row_for_export(self, ra):
        row = ra.metrics_row_for_export(
            {
                "model_name": "autointent_zero-shot-encoders_autoembedder_10shot_seed42",
                "mode": "10shot",
                "n_shots": 10,
                "seed": 42,
                "f1": 0.8,
                "precision": 0.7,
                "recall": 0.9,
                "over_refusal_rate": 0.1,
                "extra": {
                    "preset": "zero-shot-encoders",
                    "search_space_preset": "zero-shot-encoders",
                    "embedder_hf_model": "intfloat/multilingual-e5-large-instruct",
                    "model_dir": "/tmp/model",
                },
            }
        )
        assert row["preset"] == "zero-shot-encoders"
        assert row["f1"] == 0.8
        assert row["model_dir"] == "/tmp/model"
