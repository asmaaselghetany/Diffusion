"""Evaluation metrics and utilities.

Keep this package import light: Hub Fast-dLLM eval only needs code-task
helpers and must not pull torchmetrics / Lightning via ``from .metrics``.
"""

from __future__ import annotations

__all__ = ['Metrics', 'BD3Metrics']


def __getattr__(name: str):
  if name in ('Metrics', 'BD3Metrics'):
    from .metrics import BD3Metrics, Metrics
    return Metrics if name == 'Metrics' else BD3Metrics
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
