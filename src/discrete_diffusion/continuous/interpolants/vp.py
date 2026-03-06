"""VP (DDPM-like) interpolant utilities."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..math import VPSchedule, q_sample
from .base import _broadcast_mask


class VPInterpolant:
    """Discrete VP interpolant using centralized schedule math."""

    def __init__(self, schedule: VPSchedule) -> None:
        self.schedule = schedule

    def sample(self, x: Tensor, noise: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        z = q_sample(x0=x, eps=noise, t=t, sched=self.schedule)
        m = _broadcast_mask(mask, x)
        if m is not None:
            z = torch.where(m, z, x)
        return z

    def conditional_velocity(self, x: Tensor, noise: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Discrete-time VP does not have a single canonical continuous velocity.
        # We return epsilon as a practical conditioning target for VP objectives.
        del x, t
        v = noise
        m = _broadcast_mask(mask, v)
        if m is not None:
            v = torch.where(m, v, torch.zeros_like(v))
        return v


__all__ = ["VPInterpolant"]
