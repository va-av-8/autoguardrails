from __future__ import annotations

from typing import Any

from .autogluon_wrapper import AutoGluonWrapper
from .base import BaseFrameworkWrapper
from .h2o_wrapper import H2OWrapper
from .lama_wrapper import LAMAWrapper

WRAPPER_REGISTRY = {
    "autogluon": AutoGluonWrapper,
    "h2o": H2OWrapper,
    "lama": LAMAWrapper,
}


def create_wrapper(name: str, **kwargs: Any) -> BaseFrameworkWrapper:
    """Create framework wrapper by registry name."""
    key = name.lower()
    if key not in WRAPPER_REGISTRY:
        valid = ", ".join(sorted(WRAPPER_REGISTRY))
        raise ValueError(f"Unknown framework '{name}'. Expected one of: {valid}")
    return WRAPPER_REGISTRY[key](**kwargs)
