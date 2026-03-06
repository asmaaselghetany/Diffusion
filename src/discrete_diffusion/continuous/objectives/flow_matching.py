"""Rectified flow-matching objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..interpolants import RectifiedInterpolant
from .base import Objective, ObjectiveOutput, combine_masks, masked_mse


@dataclass
class FlowMatchingObjectiveConfig:
    t_min: float = 1e-4
    t_max: float = 1.0 - 1e-4


class FlowMatchingObjective(Objective):
    name = "flow_matching"

    def __init__(self, config: Optional[FlowMatchingObjectiveConfig] = None) -> None:
        self.config = config or FlowMatchingObjectiveConfig()
        self.interpolant = RectifiedInterpolant()

    def _predict_velocity(
        self,
        model,
        z_t: Tensor,
        t: Tensor,
        attention_mask: Optional[Tensor],
        span_mask: Optional[Tensor],
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        if hasattr(model, "predict_velocity"):
            return model.predict_velocity(
                z_t,
                t,
                attention_mask=attention_mask,
                span_mask=span_mask,
                self_cond=self_cond,
            )
        out = model(z_t, t, attention_mask=attention_mask, span_mask=span_mask, self_cond=self_cond)
        if isinstance(out, dict):
            v = out.get("velocity", None)
            if v is None:
                v = out.get("model_out", None)
            if v is None:
                raise RuntimeError("FlowMatchingObjective expected dict output with 'velocity' or 'model_out'")
            return v
        return out

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
        t = torch.rand(bsz, device=x0.device)
        t = t * (self.config.t_max - self.config.t_min) + self.config.t_min

        noise = torch.randn_like(x0)
        z_t = self.interpolant.sample(x=x0, noise=noise, t=t, mask=span_mask)
        v_target = self.interpolant.conditional_velocity(x=x0, noise=noise, t=t, mask=span_mask)

        self_cond = None
        if bool(getattr(model, "self_conditioning", False)):
            sc_prob = float(getattr(model, "self_conditioning_prob", 0.5))
            if torch.rand((), device=x0.device).item() < sc_prob:
                with torch.no_grad():
                    self_cond = self._predict_velocity(model, z_t, t, attention_mask, span_mask).detach()

        v_pred = self._predict_velocity(model, z_t, t, attention_mask, span_mask, self_cond=self_cond)

        loss_mask = combine_masks(attention_mask, span_mask, for_loss=True)
        main_loss = masked_mse(v_pred, v_target, loss_mask)

        x0_hat = noise - v_pred
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
            target=v_target,
        )


__all__ = ["FlowMatchingObjective", "FlowMatchingObjectiveConfig"]
