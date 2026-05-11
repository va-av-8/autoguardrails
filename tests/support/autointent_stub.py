"""
Minimal stub for `autointent` when the package is not installed (e.g. CI without GPU stack).

If the real package is importable, stubbing is skipped.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def install_autointent_stub() -> None:
    """Register lightweight autointent modules in sys.modules if import fails."""
    try:
        import autointent  # noqa: F401

        return
    except ImportError:
        pass

    if "autointent" in sys.modules:
        return

    autointent_mod = types.ModuleType("autointent")

    class _Pipeline:
        @classmethod
        def from_preset(cls, name):
            return MagicMock()

        @classmethod
        def load(cls, path):
            return MagicMock()

    class _Dataset:
        @classmethod
        def from_dict(cls, d):
            return MagicMock()

    autointent_mod.Pipeline = _Pipeline
    autointent_mod.Dataset = _Dataset
    sys.modules["autointent"] = autointent_mod

    configs_mod = types.ModuleType("autointent.configs")

    class EmbedderConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DataConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LoggingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    configs_mod.EmbedderConfig = EmbedderConfig
    configs_mod.DataConfig = DataConfig
    configs_mod.LoggingConfig = LoggingConfig
    sys.modules["autointent.configs"] = configs_mod
