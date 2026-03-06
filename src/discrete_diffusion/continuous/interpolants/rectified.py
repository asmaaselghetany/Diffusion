"""Rectified flow interpolant."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import _broadcast_mask


class RectifiedInterpolant:
    """z_t = (1 - t) x + t noise, v_cond = noise - x."""

    def sample(self, x: Tensor, noise: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        while t.ndim < x.ndim:
            t = t.unsqueeze(-1)
        z = (1.0 - t) * x + t * noise
        m = _broadcast_mask(mask, x)
        if m is not None:
            z = torch.where(m, z, x)
        return z

    def conditional_velocity(self, x: Tensor, noise: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        del t
        v = noise - x
        m = _broadcast_mask(mask, x)
        if m is not None:
            v = torch.where(m, v, torch.zeros_like(v))
        return v


__all__ = ["RectifiedInterpolant"]
