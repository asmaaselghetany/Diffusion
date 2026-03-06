"""Startup smoke checks for centralized sampler math."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..math import (
    build_vp_schedule_from_alphabar,
    eps_from_v,
    eps_from_x0,
    q_posterior,
    q_sample,
    v_from_x0_eps,
    x0_from_eps,
    x0_from_v,
)


@dataclass
class SmokeResult:
    ok: bool
    message: str



def _tiny_schedule(T: int = 16, device: Optional[torch.device] = None) -> torch.Tensor:
    # Explicitly set alphabar[0]=1.0 so the terminal clean state is reachable.
    ab = torch.linspace(1.0, 1e-3, T, dtype=torch.float32, device=device)
    ab[0] = 1.0
    return ab



def run_sampler_smoke(device: Optional[torch.device] = None) -> SmokeResult:
    try:
        dev = device or torch.device("cpu")
        sched = build_vp_schedule_from_alphabar(_tiny_schedule(device=dev))

        x0 = torch.randn(2, 5, 7, device=dev)
        eps = torch.randn_like(x0)

        timesteps = torch.tensor([0, 1, sched.T // 2, sched.T - 2, sched.T - 1], device=dev, dtype=torch.long)

        for t in timesteps:
            t_vec = torch.full((x0.shape[0],), int(t.item()), device=dev, dtype=torch.long)
            x_t = q_sample(x0=x0, eps=eps, t=t_vec, sched=sched)

            eps2 = eps_from_x0(x_t, x0, t_vec, sched)
            x02 = x0_from_eps(x_t, eps, t_vec, sched)
            v = v_from_x0_eps(x0, eps, t_vec, sched)
            x03 = x0_from_v(x_t, v, t_vec, sched)
            eps3 = eps_from_v(x_t, v, t_vec, sched)

            if int(t.item()) > 0:
                if not torch.allclose(eps2, eps, atol=1e-5, rtol=1e-5):
                    return SmokeResult(False, f"eps_from_x0 roundtrip failed at t={int(t.item())}")
            else:
                if not torch.isfinite(eps2).all():
                    return SmokeResult(False, "eps_from_x0 produced non-finite values at t=0")
            if not torch.allclose(x02, x0, atol=1e-5, rtol=1e-5):
                return SmokeResult(False, f"x0_from_eps roundtrip failed at t={int(t.item())}")
            if not torch.allclose(x03, x0, atol=1e-5, rtol=1e-5):
                return SmokeResult(False, f"x0_from_v roundtrip failed at t={int(t.item())}")
            if not torch.allclose(eps3, eps, atol=1e-5, rtol=1e-5):
                return SmokeResult(False, f"eps_from_v roundtrip failed at t={int(t.item())}")

        # Oracle reverse (DDPM posterior) should end at x0 due deterministic t=0.
        tT = torch.full((x0.shape[0],), sched.T - 1, device=dev, dtype=torch.long)
        x_t = q_sample(x0=x0, eps=torch.randn_like(x0), t=tT, sched=sched)

        for t_val in range(sched.T - 1, -1, -1):
            t_vec = torch.full((x0.shape[0],), t_val, device=dev, dtype=torch.long)
            mean, var, _ = q_posterior(x0=x0, x_t=x_t, t=t_vec, sched=sched)
            if t_val > 0:
                x_t = mean + torch.sqrt(var.clamp(min=1e-20)) * torch.randn_like(mean)
            else:
                x_t = mean

        if not torch.allclose(x_t, x0, atol=5e-5, rtol=5e-5):
            return SmokeResult(False, "oracle DDPM reverse reconstruction failed")

        return SmokeResult(True, "sampler smoke passed")
    except Exception as exc:  # pylint: disable=broad-except
        return SmokeResult(False, f"sampler smoke raised: {exc}")


__all__ = ["SmokeResult", "run_sampler_smoke"]
