"""Evaluation metrics and utilities.

This module avoids importing heavy metric dependencies at package import time.
`Metrics` and `BD3Metrics` remain available via lazy attribute access.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Metrics", "BD3Metrics"]



def __getattr__(name: str) -> Any:
    if name in __all__:
        from .metrics import BD3Metrics, Metrics

        exports = {
            "Metrics": Metrics,
            "BD3Metrics": BD3Metrics,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
