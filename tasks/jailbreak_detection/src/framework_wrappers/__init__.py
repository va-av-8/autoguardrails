"""Framework wrappers for binary Jailbreak Detection."""

from .autogluon_wrapper import AutoGluonBinaryWrapper
from .base_binary import BinaryFrameworkWrapper
from .h2o_wrapper import H2OBinaryWrapper
from .lama_wrapper import LAMABinaryWrapper

WRAPPER_REGISTRY = {
    "autogluon": AutoGluonBinaryWrapper,
    "h2o": H2OBinaryWrapper,
    "lama": LAMABinaryWrapper,
}


def create_wrapper(name: str, **kwargs) -> BinaryFrameworkWrapper:
    """Create framework wrapper by registry name."""
    key = name.lower()
    if key not in WRAPPER_REGISTRY:
        valid = ", ".join(sorted(WRAPPER_REGISTRY))
        raise ValueError(f"Unknown framework '{name}'. Expected one of: {valid}")
    return WRAPPER_REGISTRY[key](**kwargs)


__all__ = [
    "BinaryFrameworkWrapper",
    "AutoGluonBinaryWrapper",
    "H2OBinaryWrapper",
    "LAMABinaryWrapper",
    "create_wrapper",
    "WRAPPER_REGISTRY",
]
