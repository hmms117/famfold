"""Legacy namespace for experiment utilities.

This package provides compatibility shims for modules that were relocated to
``hypothesis_test``. Direct imports should use the new package instead.
"""

from importlib import import_module
from typing import Any
import sys

__all__ = ["hypothesis_test"]

# Preload the relocated module so ``import experiments.hypothesis_test`` works
# when consumers rely on the historical package path.
_module = import_module("hypothesis_test")
sys.modules[__name__ + ".hypothesis_test"] = _module


def __getattr__(name: str) -> Any:
    if name == "hypothesis_test":
        return _module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)
