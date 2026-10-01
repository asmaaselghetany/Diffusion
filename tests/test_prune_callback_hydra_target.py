"""Regression: package __init__ must not shadow the prune submodule.

Hydra resolves
``discrete_diffusion.callbacks.prune_periodic_checkpoints.PrunePeriodicCheckpoints``.
Re-exporting the helper *function* ``prune_periodic_checkpoints`` from
``callbacks/__init__.py`` binds that name on the package to a function, so
Hydra sees a function (no ``PrunePeriodicCheckpoints`` attr) and every
block_qwen train InstantiationException's (jobs 1968980 / 1969453 / 1968017).
"""

from __future__ import annotations

import discrete_diffusion.callbacks as callbacks_pkg
import discrete_diffusion.callbacks.prune_periodic_checkpoints as prune_mod
from discrete_diffusion.callbacks.prune_periodic_checkpoints import (
    PrunePeriodicCheckpoints,
)


def test_prune_submodule_not_shadowed_by_helper_function():
  attr = getattr(callbacks_pkg, 'prune_periodic_checkpoints')
  assert attr is prune_mod
  assert hasattr(attr, 'PrunePeriodicCheckpoints')
  assert attr.PrunePeriodicCheckpoints is PrunePeriodicCheckpoints


def test_hydra_get_class_resolves_prune_callback():
  from hydra.utils import get_class

  cls = get_class(
      'discrete_diffusion.callbacks.prune_periodic_checkpoints.'
      'PrunePeriodicCheckpoints')
  assert cls is PrunePeriodicCheckpoints
