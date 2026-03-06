"""DDPM-style objective for continuous embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..math import (
    VPSchedule,
    q_sample,
    x0_from_eps,
    eps_from_x0,
    x0_from_v,
    v_from_x0_eps,
)
from .base import Objective, ObjectiveOutput, combine_masks, masked_mse


@dataclass
class DDPMObjectiveConfig:
    parameterization: str = "x0"  # x0 | epsilon | v
    min_snr_gamma: Optional[float] = None


class DDPMObjective(Objective):
    name = "ddpm"

    def __init__(self, schedule: VPSchedule, config: Optional[DDPMObjectiveConfig] = None) -> None:
        self.schedule = schedule
        self.config = config or DDPMObjectiveConfig()

    def _sample_t(self, bsz: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.schedule.T, (bsz,), device=device, dtype=torch.long)

    def _time_to_model(self, t: Tensor) -> Tensor:
        if self.schedule.T <= 1:
            return torch.zeros_like(t, dtype=torch.float32)
        return t.float() / float(self.schedule.T - 1)

    def _compute_target(self, x0: Tensor, eps: Tensor, t: Tensor, x_t: Tensor) -> Tensor:
        p = self.config.parameterization
        if p == "x0":
            return x0
        if p == "epsilon":
            return eps
        if p == "v":
            return v_from_x0_eps(x0, eps, t, self.schedule)
        raise ValueError(f"Unsupported DDPM parameterization: {p}")

    def _to_x0(self, model_out: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        p = self.config.parameterization
        if p == "x0":
            return model_out
        if p == "epsilon":
            return x0_from_eps(x_t, model_out, t, self.schedule)
        if p == "v":
            return x0_from_v(x_t, model_out, t, self.schedule)
        raise ValueError(f"Unsupported DDPM parameterization: {p}")

    def _min_snr_weight(self, t: Tensor, dtype: torch.dtype, device: torch.device) -> Optional[Tensor]:
        gamma = self.config.min_snr_gamma
        if gamma is None:
            return None
        # Keep schedule lookup on the timestep device first to avoid CPU/CUDA index mismatches.
        ab_table = self.schedule.alphabar
        if ab_table.device != t.device:
            ab_table = ab_table.to(device=t.device)
        ab = ab_table.index_select(0, t.view(-1)).to(device=device, dtype=torch.float32)
        snr = ab / (1.0 - ab).clamp(min=1e-20)
        w = torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr.clamp(min=1e-20)
        return w.to(dtype=dtype)

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
        t = self._sample_t(bsz, x0.device)
        t_model = self._time_to_model(t)

        eps = torch.randn_like(x0)
        z_t = q_sample(x0=x0, eps=eps, t=t, sched=self.schedule)

        if span_mask is not None:
            mask = span_mask.bool()
            while mask.ndim < z_t.ndim:
                mask = mask.unsqueeze(-1)
            z_t = torch.where(mask, z_t, x0)

        target = self._compute_target(x0=x0, eps=eps, t=t, x_t=z_t)

        self_cond = None
        if bool(getattr(model, "self_conditioning", False)):
            sc_prob = float(getattr(model, "self_conditioning_prob", 0.5))
            if torch.rand((), device=x0.device).item() < sc_prob:
                with torch.no_grad():
                    first = model(z_t, t_model, attention_mask=attention_mask, span_mask=span_mask)
                    if isinstance(first, dict):
                        first = first.get("model_out", first.get("x", None))
                        if first is None:
                            raise RuntimeError("DDPMObjective self-conditioning expected dict with model output")
                    self_cond = self._to_x0(model_out=first, x_t=z_t, t=t).detach()

        pred = model(
            z_t,
            t_model,
            attention_mask=attention_mask,
            span_mask=span_mask,
            self_cond=self_cond,
        )
        if isinstance(pred, dict):
            pred = pred.get("model_out", pred.get("x", None))
            if pred is None:
                raise RuntimeError("DDPMObjective expected tensor output or dict with 'model_out'")

        loss_mask = combine_masks(attention_mask, span_mask, for_loss=True)
        w = self._min_snr_weight(t, dtype=pred.dtype, device=pred.device)
        if w is not None:
            while w.ndim < pred.ndim:
                w = w.unsqueeze(-1)
            pred_w = pred * torch.sqrt(w)
            target_w = target * torch.sqrt(w)
            main_loss = masked_mse(pred=pred_w, target=target_w, mask=loss_mask)
        else:
            main_loss = masked_mse(pred=pred, target=target, mask=loss_mask)

        x0_hat = self._to_x0(model_out=pred, x_t=z_t, t=t)

        return ObjectiveOutput(loss=main_loss, main_loss=main_loss, x0_hat=x0_hat, z_t=z_t, target=target)


__all__ = ["DDPMObjective", "DDPMObjectiveConfig"]
