"""Rectified flow / mean-flow samplers."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class RectifiedSampler:
    """Sampling updates for one-step and few-step rectified generation."""

    _KNOWN_TOL = 1e-5

    @staticmethod
    def one_step(*, noise: Tensor, u: Tensor, x_known: Optional[Tensor] = None, span_mask: Optional[Tensor] = None) -> Tensor:
        x_hat = noise - u
        if span_mask is not None and x_known is not None:
            m = span_mask.bool()
            while m.ndim < x_hat.ndim:
                m = m.unsqueeze(-1)
            x_hat = torch.where(m, x_hat, x_known)
            known_drift = (x_hat - x_known).abs().masked_fill(m, 0.0).max()
            if known_drift.item() > RectifiedSampler._KNOWN_TOL:
                raise RuntimeError(
                    "Rectified one-step sampler modified known tokens beyond tolerance: "
                    f"{known_drift.item():.3e}"
                )
        return x_hat

    @staticmethod
    def few_step_update(
        z_tk: Tensor,
        u_tk: Tensor,
        t_k: Tensor,
        t_km1: Tensor,
        *,
        x_known: Optional[Tensor] = None,
        span_mask: Optional[Tensor] = None,
    ) -> Tensor:
        dt = (t_k - t_km1)
        while dt.ndim < z_tk.ndim:
            dt = dt.unsqueeze(-1)
        z_next = z_tk - dt * u_tk
        if span_mask is not None and x_known is not None:
            m = span_mask.bool()
            while m.ndim < z_next.ndim:
                m = m.unsqueeze(-1)
            z_next = torch.where(m, z_next, x_known)
            known_drift = (z_next - x_known).abs().masked_fill(m, 0.0).max()
            if known_drift.item() > RectifiedSampler._KNOWN_TOL:
                raise RuntimeError(
                    "Rectified few-step sampler modified known tokens beyond tolerance: "
                    f"{known_drift.item():.3e}"
                )
        return z_next


__all__ = ["RectifiedSampler"]
