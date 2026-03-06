"""Centralized VP-diffusion math utilities.

This module is the single source of truth for:
- VP schedules (betas, alphas, alphabar, sigmas)
- parameterization conversions (x0/eps/v)
- posterior coefficients for q(x_{t-1} | x_t, x0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


_FLOAT_CLAMP_MIN = 1e-20


@dataclass(frozen=True)
class VPSchedule:
    """Discrete VP schedule tensors.

    Shapes:
      - all fields are `[T]`
    """

    betas: Tensor
    alphas: Tensor
    alphabar: Tensor
    sqrt_ab: Tensor
    sqrt_1mab: Tensor
    sigmas: Tensor
    posterior_mean_coef1: Tensor
    posterior_mean_coef2: Tensor
    posterior_variance: Tensor
    posterior_log_variance: Tensor

    @property
    def T(self) -> int:
        return int(self.betas.shape[0])



def _assert_schedule_1d(x: Tensor, name: str) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D [T], got shape {tuple(x.shape)}")



def _validate_timestep(t: Tensor, T: int) -> None:
    if t.numel() == 0:
        return
    t_min = int(t.min().item())
    t_max = int(t.max().item())
    if t_min < 0 or t_max >= T:
        raise ValueError(f"Invalid timestep range [{t_min}, {t_max}] for T={T}")



def _as_timestep_tensor(t: int | Tensor, device: torch.device) -> Tensor:
    if isinstance(t, int):
        return torch.tensor([t], device=device, dtype=torch.long)
    if not torch.is_tensor(t):
        raise TypeError(f"Unsupported timestep type: {type(t)}")
    if t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"Timestep tensor must be integer dtype, got {t.dtype}")
    return t.to(device=device, dtype=torch.long)



def _extract(values: Tensor, t: int | Tensor, ref: Tensor) -> Tensor:
    """Gather 1D schedule values at timestep(s) and broadcast to `ref` shape."""
    _assert_schedule_1d(values, "values")
    if values.device != ref.device:
        values = values.to(ref.device)
    t_idx = _as_timestep_tensor(t, ref.device)
    _validate_timestep(t_idx, int(values.shape[0]))

    gathered = values.index_select(0, t_idx.view(-1)).view(*t_idx.shape)
    while gathered.ndim < ref.ndim:
        gathered = gathered.unsqueeze(-1)
    return gathered.to(dtype=ref.dtype)


def _math_dtype(dtype: torch.dtype) -> torch.dtype:
    # Keep schedule algebra stable for low-precision tensors.
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype



def build_vp_schedule(betas: Tensor, clamp_min: float = _FLOAT_CLAMP_MIN) -> VPSchedule:
    """Build a discrete VP schedule from DDPM betas."""
    _assert_schedule_1d(betas, "betas")
    if betas.numel() < 2:
        raise ValueError("VP schedule requires at least 2 timesteps")
    if not torch.is_floating_point(betas):
        betas = betas.float()

    if (betas < 0).any() or (betas >= 1).any():
        raise ValueError("betas must satisfy 0 <= beta_t < 1")

    alphas = 1.0 - betas
    alphabar = torch.cumprod(alphas, dim=0)

    if (alphabar <= 0).any() or (alphabar > 1).any():
        raise ValueError("alphabar must satisfy 0 < alphabar_t <= 1")

    sqrt_ab = torch.sqrt(alphabar.clamp(min=clamp_min))
    sqrt_1mab = torch.sqrt((1.0 - alphabar).clamp(min=clamp_min))
    sigmas = torch.sqrt(betas.clamp(min=clamp_min))

    # q(x_{t-1}|x_t,x0)
    # Note: for t=0 we define deterministic end: mean=x0, var=0.
    alphabar_prev = torch.cat([torch.ones_like(alphabar[:1]), alphabar[:-1]], dim=0)

    posterior_variance = betas * (1.0 - alphabar_prev) / (1.0 - alphabar).clamp(min=clamp_min)
    posterior_variance = posterior_variance.clamp(min=0.0)
    posterior_variance[0] = 0.0

    posterior_log_variance = torch.log(posterior_variance.clamp(min=clamp_min))

    posterior_mean_coef1 = betas * torch.sqrt(alphabar_prev.clamp(min=clamp_min)) / (1.0 - alphabar).clamp(min=clamp_min)
    posterior_mean_coef2 = (1.0 - alphabar_prev) * torch.sqrt(alphas.clamp(min=clamp_min)) / (1.0 - alphabar).clamp(min=clamp_min)

    posterior_mean_coef1[0] = 1.0
    posterior_mean_coef2[0] = 0.0

    return VPSchedule(
        betas=betas,
        alphas=alphas,
        alphabar=alphabar,
        sqrt_ab=sqrt_ab,
        sqrt_1mab=sqrt_1mab,
        sigmas=sigmas,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        posterior_variance=posterior_variance,
        posterior_log_variance=posterior_log_variance,
    )



def build_vp_schedule_from_alphabar(alphabar: Tensor, clamp_min: float = _FLOAT_CLAMP_MIN) -> VPSchedule:
    """Build a VP schedule from precomputed alphabar[t]."""
    _assert_schedule_1d(alphabar, "alphabar")
    if alphabar.numel() < 2:
        raise ValueError("alphabar schedule requires at least 2 timesteps")

    if not torch.is_floating_point(alphabar):
        alphabar = alphabar.float()

    if (alphabar <= 0).any() or (alphabar > 1).any():
        raise ValueError("alphabar must satisfy 0 < alphabar_t <= 1")

    alphas = torch.empty_like(alphabar)
    alphas[0] = alphabar[0]
    alphas[1:] = (alphabar[1:] / alphabar[:-1].clamp(min=clamp_min)).clamp(min=clamp_min, max=1.0)
    betas = (1.0 - alphas).clamp(min=clamp_min, max=1.0 - clamp_min)
    return build_vp_schedule(betas=betas, clamp_min=clamp_min)


def scale_alphabar_noise(alphabar: Tensor, noise_scale: float, clamp_min: float = _FLOAT_CLAMP_MIN) -> Tensor:
    """Scale the forward noise level while keeping alphabar in (0, 1]."""
    if not torch.is_floating_point(alphabar):
        alphabar = alphabar.float()
    scale = float(noise_scale)
    if scale <= 0:
        raise ValueError(f"noise_scale must be > 0, got {scale}")
    if abs(scale - 1.0) < 1e-12:
        return alphabar
    one_minus = (1.0 - alphabar) * scale
    one_minus = one_minus.clamp(min=0.0, max=1.0 - clamp_min)
    return (1.0 - one_minus).clamp(min=clamp_min, max=1.0)



def x0_from_eps(x_t: Tensor, eps: Tensor, t: int | Tensor, sched: VPSchedule) -> Tensor:
    out_dtype = x_t.dtype
    mdtype = _math_dtype(out_dtype)
    x_t_m = x_t.to(mdtype)
    eps_m = eps.to(mdtype)
    a = _extract(sched.sqrt_ab, t, x_t_m)
    b = _extract(sched.sqrt_1mab, t, x_t_m)
    out = (x_t_m - b * eps_m) / a.clamp(min=_FLOAT_CLAMP_MIN)
    return out.to(out_dtype)



def eps_from_x0(x_t: Tensor, x0: Tensor, t: int | Tensor, sched: VPSchedule) -> Tensor:
    out_dtype = x_t.dtype
    mdtype = _math_dtype(out_dtype)
    x_t_m = x_t.to(mdtype)
    x0_m = x0.to(mdtype)
    a = _extract(sched.sqrt_ab, t, x_t_m)
    b = _extract(sched.sqrt_1mab, t, x_t_m)
    out = (x_t_m - a * x0_m) / b.clamp(min=_FLOAT_CLAMP_MIN)
    return out.to(out_dtype)



def v_from_x0_eps(x0: Tensor, eps: Tensor, t: int | Tensor, sched: VPSchedule) -> Tensor:
    out_dtype = x0.dtype
    mdtype = _math_dtype(out_dtype)
    x0_m = x0.to(mdtype)
    eps_m = eps.to(mdtype)
    a = _extract(sched.sqrt_ab, t, x0_m)
    b = _extract(sched.sqrt_1mab, t, x0_m)
    # v = sqrt(alphabar_t) * eps - sqrt(1 - alphabar_t) * x0
    out = a * eps_m - b * x0_m
    return out.to(out_dtype)



def x0_from_v(x_t: Tensor, v: Tensor, t: int | Tensor, sched: VPSchedule) -> Tensor:
    out_dtype = x_t.dtype
    mdtype = _math_dtype(out_dtype)
    x_t_m = x_t.to(mdtype)
    v_m = v.to(mdtype)
    a = _extract(sched.sqrt_ab, t, x_t_m)
    b = _extract(sched.sqrt_1mab, t, x_t_m)
    # Inverse of [x_t, v] rotation.
    out = a * x_t_m - b * v_m
    return out.to(out_dtype)



def eps_from_v(x_t: Tensor, v: Tensor, t: int | Tensor, sched: VPSchedule) -> Tensor:
    out_dtype = x_t.dtype
    mdtype = _math_dtype(out_dtype)
    x_t_m = x_t.to(mdtype)
    v_m = v.to(mdtype)
    a = _extract(sched.sqrt_ab, t, x_t_m)
    b = _extract(sched.sqrt_1mab, t, x_t_m)
    # Inverse of [x_t, v] rotation.
    out = b * x_t_m + a * v_m
    return out.to(out_dtype)



def q_posterior(x0: Tensor, x_t: Tensor, t: int | Tensor, sched: VPSchedule) -> Tuple[Tensor, Tensor, Tensor]:
    """Return mean, var, logvar of q(x_{t-1}|x_t,x0)."""
    out_dtype = x_t.dtype
    mdtype = _math_dtype(out_dtype)
    x0_m = x0.to(mdtype)
    x_t_m = x_t.to(mdtype)
    coef1 = _extract(sched.posterior_mean_coef1, t, x_t_m)
    coef2 = _extract(sched.posterior_mean_coef2, t, x_t_m)
    mean = coef1 * x0_m + coef2 * x_t_m

    var = _extract(sched.posterior_variance, t, x_t_m)
    logvar = _extract(sched.posterior_log_variance, t, x_t_m)
    return mean.to(out_dtype), var.to(out_dtype), logvar.to(out_dtype)



def q_sample(x0: Tensor, eps: Tensor, t: int | Tensor, sched: VPSchedule) -> Tensor:
    out_dtype = x0.dtype
    mdtype = _math_dtype(out_dtype)
    x0_m = x0.to(mdtype)
    eps_m = eps.to(mdtype)
    a = _extract(sched.sqrt_ab, t, x0_m)
    b = _extract(sched.sqrt_1mab, t, x0_m)
    out = a * x0_m + b * eps_m
    return out.to(out_dtype)


__all__ = [
    "VPSchedule",
    "build_vp_schedule",
    "build_vp_schedule_from_alphabar",
    "scale_alphabar_noise",
    "q_sample",
    "x0_from_eps",
    "eps_from_x0",
    "v_from_x0_eps",
    "x0_from_v",
    "eps_from_v",
    "q_posterior",
]
