"""Interpolant interfaces for continuous objectives."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Interpolant(Protocol):
    def sample(self, x: Tensor, noise: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Return z_t."""

    def conditional_velocity(self, x: Tensor, noise: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Return v_cond = d/dt z_t."""



def _broadcast_mask(mask: Optional[Tensor], ref: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    m = mask.to(device=ref.device, dtype=torch.bool)
    while m.ndim < ref.ndim:
        m = m.unsqueeze(-1)
    return m


__all__ = ["Interpolant", "_broadcast_mask"]
