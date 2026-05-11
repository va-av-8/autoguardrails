"""Shared pytest configuration for autoguardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.autointent_stub import install_autointent_stub

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config: pytest.Config) -> None:
    _ = config
    install_autointent_stub()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT
