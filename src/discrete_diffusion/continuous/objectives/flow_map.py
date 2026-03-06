"""Flow-map objective: train F_θ(z_t, t, s) → z_s directly.

Instead of predicting velocities (flow matching) or mean-flow + JVP (iMF),
the model learns a *flow map* that directly transports z_t to z_s for
arbitrary time pairs (t, s) with s < t.

Training:
    1. Sample (t, s) with s < t from [t_min, t_max]
    2. Create z_t, z_s from the same (x_0, noise) pair — same ODE trajectory
    3. F_θ(z_t, t, s) predicts z_s
    4. Loss = MSE(F_θ(z_t, t, s), z_s)

Architecture reuse:
    The model's default forward (parameterization='x0') with residual
    connection is: F(z_t, t, s) = net_θ(z_t, t, r=s) + z_t.
    At init (zero-init), this is the identity map (F = z_t).
    The r_embedder provides the s-conditioning.

Relationship to other objectives:
    - flow_matching: learns instantaneous velocity v(z, t); Euler-integrated
    - iMF/meanflow: learns average velocity u(z, r, t) with JVP correction
    - flow_map (this): learns position F(z, t, s) directly — no JVP needed.
      Setting s=0 always recovers standard x0-prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..interpolants import RectifiedInterpolant
from .base import Objective, ObjectiveOutput, combine_masks, masked_mse


@dataclass
class FlowMapObjectiveConfig:
    dt_min: float = 1e-3
    t_min: float = 1e-4
    t_max: float = 1.0 - 1e-4
    s_zero_prob: float = 0.0


class FlowMapObjective(Objective):
    name = "flow_map"

    def __init__(self, config: Optional[FlowMapObjectiveConfig] = None) -> None:
        self.config = config or FlowMapObjectiveConfig()
        self.interpolant = RectifiedInterpolant()

    # ------------------------------------------------------------------
    # Time-pair sampling
    # ------------------------------------------------------------------

    def _sample_t_s(self, bsz: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Sample (t, s) pairs with 0 <= s < t and t - s >= dt_min."""
        cfg = self.config
        effective_t_min = max(cfg.t_min, cfg.dt_min)
        t = torch.rand(bsz, device=device) * (cfg.t_max - effective_t_min) + effective_t_min

        max_s = (t - cfg.dt_min).clamp(min=0.0)
        s = torch.rand(bsz, device=device) * max_s

        if cfg.s_zero_prob > 0.0:
            zero_mask = torch.rand(bsz, device=device) < cfg.s_zero_prob
            s = torch.where(zero_mask, torch.zeros_like(s), s)

        s = torch.minimum(s, (t - cfg.dt_min).clamp(min=0.0))
        return t, s

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def compute(
        self,
        *,
        model,
        x0: Tensor,
        attention_mask: Optional[Tensor] = None,
        span_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        del input_ids
        bsz = x0.shape[0]
        t, s = self._sample_t_s(bsz, x0.device)

        noise = torch.randn_like(x0)
        z_t = self.interpolant.sample(x=x0, noise=noise, t=t, mask=span_mask)
        z_s = self.interpolant.sample(x=x0, noise=noise, t=s, mask=span_mask)

        # Optional self-conditioning
        self_cond = None
        if bool(getattr(model, "self_conditioning", False)):
            sc_prob = float(getattr(model, "self_conditioning_prob", 0.5))
            if torch.rand((), device=x0.device).item() < sc_prob:
                with torch.no_grad():
                    self_cond = model(
                        z_t, t, r=s,
                        attention_mask=attention_mask,
                        span_mask=span_mask,
                    ).detach()

        # F_θ(z_t, t, s): default forward with residual — predicts z_s
        z_s_hat = model(
            z_t, t, r=s,
            attention_mask=attention_mask,
            span_mask=span_mask,
            self_cond=self_cond,
        )

        loss_mask = combine_masks(attention_mask, span_mask, for_loss=True)
        main_loss = masked_mse(z_s_hat, z_s, loss_mask)

        # x0_hat recovery: z_s = (1-s)*x0 + s*noise  =>  x0 = (z_s - s*noise)/(1-s)
        s_b = s.clone()
        while s_b.ndim < x0.ndim:
            s_b = s_b.unsqueeze(-1)
        x0_hat = (z_s_hat - s_b * noise) / (1.0 - s_b).clamp(min=1e-6)

        if span_mask is not None:
            m = span_mask.bool()
            while m.ndim < x0.ndim:
                m = m.unsqueeze(-1)
            x0_hat = torch.where(m, x0_hat, x0)

        return ObjectiveOutput(
            loss=main_loss,
            main_loss=main_loss,
            x0_hat=x0_hat,
            z_t=z_t,
            target=z_s,
        )


__all__ = ["FlowMapObjective", "FlowMapObjectiveConfig"]
