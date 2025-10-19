"""Compatibility shim that re-exports the relocated module."""

from hypothesis_test import cli as _impl

from hypothesis_test.cli import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [name for name in dir(_impl) if not name.startswith("_")])
__doc__ = _impl.__doc__
