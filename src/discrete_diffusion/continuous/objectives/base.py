"""Objective interfaces for continuous embedding diffusion/flow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class ObjectiveOutput:
    loss: Tensor
    main_loss: Tensor
    aux_losses: Dict[str, Tensor] = field(default_factory=dict)
    x0_hat: Optional[Tensor] = None
    z_t: Optional[Tensor] = None
    target: Optional[Tensor] = None


class Objective(ABC):
    name: str

    @abstractmethod
    def compute(
        self,
        *,
        model,
        x0: Tensor,
        attention_mask: Optional[Tensor] = None,
        span_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        raise NotImplementedError



def masked_mean(tensor: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return tensor.mean()
    m = mask.to(dtype=tensor.dtype, device=tensor.device)
    while m.ndim < tensor.ndim:
        m = m.unsqueeze(-1)
    # Normalize by the full number of valid elements (including embedding dims),
    # not just valid tokens.
    m = torch.broadcast_to(m, tensor.shape)
    denom = m.sum().clamp(min=1.0)
    return (tensor * m).sum() / denom



def masked_mse(pred: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
    return masked_mean((pred - target) ** 2, mask)



def combine_masks(attention_mask: Optional[Tensor], span_mask: Optional[Tensor], *, for_loss: bool = True) -> Optional[Tensor]:
    if attention_mask is None and span_mask is None:
        return None
    if attention_mask is None:
        return span_mask
    if span_mask is None:
        return attention_mask.bool() if for_loss else attention_mask
    if for_loss:
        return attention_mask.bool() & span_mask.bool()
    return span_mask.bool()


__all__ = [
    "Objective",
    "ObjectiveOutput",
    "masked_mean",
    "masked_mse",
    "combine_masks",
]
