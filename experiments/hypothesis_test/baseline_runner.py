"""Compatibility shim that re-exports the relocated module."""

from hypothesis_test import baseline_runner as _impl

from hypothesis_test.baseline_runner import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [name for name in dir(_impl) if not name.startswith("_")])
__doc__ = _impl.__doc__
