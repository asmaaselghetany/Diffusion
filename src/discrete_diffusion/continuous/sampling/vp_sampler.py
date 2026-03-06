"""VP sampler helpers built on centralized math."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..math import (
    VPSchedule,
    eps_from_x0,
    q_posterior,
)


class VPSampler:
    def __init__(self, schedule: VPSchedule) -> None:
        self.schedule = schedule

    def ddpm_step(self, x_t: Tensor, x0_hat: Tensor, t: Tensor, add_noise: bool = True) -> Tensor:
        mean, var, _ = q_posterior(x0=x0_hat, x_t=x_t, t=t, sched=self.schedule)
        if add_noise:
            return mean + torch.sqrt(var.clamp(min=1e-20)) * torch.randn_like(mean)
        return mean

    def ddim_step(self, x_t: Tensor, x0_hat: Tensor, t: Tensor, t_prev: Tensor, eta: float = 0.0) -> Tensor:
        eps = eps_from_x0(x_t=x_t, x0=x0_hat, t=t, sched=self.schedule)

        ab_t = self.schedule.alphabar.index_select(0, t.view(-1)).to(device=x_t.device, dtype=x_t.dtype)
        ab_prev = self.schedule.alphabar.index_select(0, t_prev.view(-1)).to(device=x_t.device, dtype=x_t.dtype)
        while ab_t.ndim < x_t.ndim:
            ab_t = ab_t.unsqueeze(-1)
            ab_prev = ab_prev.unsqueeze(-1)

        sigma = float(eta) * torch.sqrt(
            ((1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-20))
            * (1.0 - (ab_t / ab_prev.clamp(min=1e-20)))
        )
        dir_xt = torch.sqrt((1.0 - ab_prev - sigma ** 2).clamp(min=0.0)) * eps
        x_prev = torch.sqrt(ab_prev.clamp(min=1e-20)) * x0_hat + dir_xt
        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_prev)
        return x_prev


__all__ = ["VPSampler"]
