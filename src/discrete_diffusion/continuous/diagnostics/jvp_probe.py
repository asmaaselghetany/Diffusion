"""JVP support probes for transformer blocks used by iMF."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class JVPProbeResult:
    ok: bool
    message: str



def _tiny_inputs(device: torch.device, dim: int = 64) -> tuple[Tensor, Tensor, Tensor]:
    z = torch.randn(1, 4, dim, device=device)
    r = torch.tensor([0.25], device=device)
    t = torch.tensor([0.75], device=device)
    return z, r, t



def probe_model_jvp(model) -> JVPProbeResult:
    """Probe whether `model.predict_u` can execute through torch.func.jvp."""
    if not hasattr(model, "predict_u"):
        return JVPProbeResult(True, "model has no predict_u; skipping JVP probe")

    try:
        device = next(model.parameters()).device
        z, r, t = _tiny_inputs(device=device, dim=getattr(model, "embed_dim", 64))

        def fn(z_in: Tensor, r_in: Tensor, t_in: Tensor) -> Tensor:
            return model.predict_u(z_in, r_in, t_in).float()

        with torch.amp.autocast(device_type=device.type if device.type in {"cuda", "cpu"} else "cuda", enabled=False):
            torch.func.jvp(
                fn,
                (z.float(), r.float(), t.float()),
                (torch.randn_like(z).float(), torch.zeros_like(r).float(), torch.ones_like(t).float()),
            )

        return JVPProbeResult(True, "jvp probe passed")
    except RuntimeError as exc:
        return JVPProbeResult(False, f"jvp probe failed: {exc}")


__all__ = ["JVPProbeResult", "probe_model_jvp"]
