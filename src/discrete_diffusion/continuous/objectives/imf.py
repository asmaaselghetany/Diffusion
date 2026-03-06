"""Improved Mean Flow objective with JVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..interpolants import RectifiedInterpolant
from .base import Objective, ObjectiveOutput, combine_masks, masked_mse


@dataclass
class IMFObjectiveConfig:
    dt_min: float = 1e-3
    t_min: float = 1e-4
    t_max: float = 1.0 - 1e-4
    stopgrad_dudt: bool = True
    use_aux_v_head: bool = True
    alpha_imf: float = 1.0
    fm_aux_weight: float = 0.05


class IMFObjective(Objective):
    name = "imf"

    def __init__(self, config: Optional[IMFObjectiveConfig] = None) -> None:
        self.config = config or IMFObjectiveConfig()
        self.interpolant = RectifiedInterpolant()

    def _predict_u(
        self,
        model,
        z_t: Tensor,
        r: Tensor,
        t: Tensor,
        attention_mask: Optional[Tensor],
        span_mask: Optional[Tensor],
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        if hasattr(model, "predict_u"):
            return model.predict_u(
                z_t,
                r,
                t,
                attention_mask=attention_mask,
                span_mask=span_mask,
                self_cond=self_cond,
            )
        out = model(z_t, t, r=r, attention_mask=attention_mask, span_mask=span_mask, self_cond=self_cond)
        if isinstance(out, dict):
            u = out.get("u", None)
            if u is None:
                u = out.get("model_out", None)
            if u is None:
                raise RuntimeError("IMFObjective expected dict output with 'u' or 'model_out'")
            return u
        return out

    def _predict_v(
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
                raise RuntimeError("IMFObjective expected dict output with 'velocity' or 'model_out'")
            return v
        return out

    def _sample_r_t(self, bsz: int, device: torch.device) -> tuple[Tensor, Tensor]:
        t = torch.rand(bsz, device=device)
        t = t * (self.config.t_max - self.config.t_min) + self.config.t_min
        # r in [0, t - dt_min)
        max_r = (t - self.config.dt_min).clamp(min=0.0)
        r = torch.rand_like(t) * max_r
        # enforce t-r >= dt_min
        r = torch.minimum(r, (t - self.config.dt_min).clamp(min=0.0))
        return r, t

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
        r, t = self._sample_r_t(bsz, x0.device)

        noise = torch.randn_like(x0)
        z_t = self.interpolant.sample(x=x0, noise=noise, t=t, mask=span_mask)
        v_cond = self.interpolant.conditional_velocity(x=x0, noise=noise, t=t, mask=span_mask)

        self_cond = None
        if bool(getattr(model, "self_conditioning", False)):
            sc_prob = float(getattr(model, "self_conditioning_prob", 0.5))
            if torch.rand((), device=x0.device).item() < sc_prob:
                with torch.no_grad():
                    if self.config.use_aux_v_head:
                        self_cond = self._predict_v(model, z_t, t, attention_mask, span_mask).detach()
                    else:
                        self_cond = self._predict_u(model, z_t, t, t, attention_mask, span_mask).detach()

        if self.config.use_aux_v_head:
            v_hat = self._predict_v(model, z_t, t, attention_mask, span_mask, self_cond=self_cond)
        else:
            v_hat = self._predict_u(model, z_t, t, t, attention_mask, span_mask, self_cond=self_cond)

        dt = (t - r).clamp(min=self.config.dt_min)

        # JVP in fp32, with autocast disabled, to avoid mixed-precision forward-mode issues.
        z_fp32 = z_t.float()
        r_fp32 = r.float()
        t_fp32 = t.float()
        v_hat_fp32 = v_hat.float()

        def fn(z_in: Tensor, r_in: Tensor, t_in: Tensor) -> Tensor:
            out = self._predict_u(model, z_in, r_in, t_in, attention_mask, span_mask, self_cond=self_cond)
            return out.float()

        device_type = x0.device.type if x0.device.type in {"cuda", "cpu"} else "cuda"
        try:
            with torch.amp.autocast(device_type=device_type, enabled=False):
                u_fp32, dudt_fp32 = torch.func.jvp(
                    fn,
                    (z_fp32, r_fp32, t_fp32),
                    (v_hat_fp32, torch.zeros_like(r_fp32), torch.ones_like(t_fp32)),
                )
        except RuntimeError as exc:
            msg = str(exc)
            raise RuntimeError(
                "torch.func.jvp failed in IMFObjective. "
                "This usually means a forward-mode AD unsupported operator is present. "
                f"Original error: {msg}"
            ) from exc

        u = u_fp32.to(dtype=x0.dtype)
        dudt = dudt_fp32.to(dtype=x0.dtype)

        dt_b = dt
        while dt_b.ndim < u.ndim:
            dt_b = dt_b.unsqueeze(-1)

        if self.config.stopgrad_dudt:
            dudt_used = dudt.detach()
        else:
            dudt_used = dudt

        V = u + dt_b * dudt_used

        loss_mask = combine_masks(attention_mask, span_mask, for_loss=True)
        l_imf = masked_mse(V, v_cond, loss_mask)
        l_fm = masked_mse(v_hat, v_cond, loss_mask)

        alpha = float(max(min(self.config.alpha_imf, 1.0), 0.0))
        total = (1.0 - alpha) * l_fm + alpha * l_imf + float(self.config.fm_aux_weight) * l_fm

        x0_hat = noise - V
        if span_mask is not None:
            m = span_mask.bool()
            while m.ndim < x0.ndim:
                m = m.unsqueeze(-1)
            x0_hat = torch.where(m, x0_hat, x0)

        return ObjectiveOutput(
            loss=total,
            main_loss=l_imf,
            aux_losses={"imf": l_imf, "fm": l_fm},
            x0_hat=x0_hat,
            z_t=z_t,
            target=v_cond,
        )


__all__ = ["IMFObjective", "IMFObjectiveConfig"]
