"""Compatibility layer for the relocated hypothesis test package."""

from importlib import import_module
from typing import Any, Dict

_impl = import_module("hypothesis_test")

def __getattr__(name: str) -> Any:
    try:
        return getattr(_impl, name)
    except AttributeError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc


def __dir__() -> list[str]:
    return sorted(__all__)

__all__ = getattr(_impl, "__all__", [name for name in dir(_impl) if not name.startswith("_")])

# Populate the module globals so attribute access behaves like the original
_globals: Dict[str, Any] = {
    name: getattr(_impl, name)
    for name in __all__
}
globals().update(_globals)

# Ensure the module reports the same documentation reference.
__doc__ = _impl.__doc__
