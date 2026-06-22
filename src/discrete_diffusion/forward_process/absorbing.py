"""Absorbing-state forward processes."""

from __future__ import annotations

import torch
from .utils import _mask_token_id, _unsqueeze
from .base import ForwardProcess
from ..noise_schedules.base import NoiseSchedule


class AbsorbingForwardProcess(ForwardProcess):
  """Absorbing-state forward process.

  Replaces tokens with the mask token with probability `(1 - alpha_t)`.
  Returns the noised ids and the per-position mask probability `p_mask`.
  """

  def __init__(self, tokenizer, schedule: NoiseSchedule, name: str | None = None) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.mask_id = _mask_token_id(tokenizer)

  @torch.no_grad()
  def corrupt_with_mask(self, input_ids: torch.Tensor, move_mask: torch.Tensor):
    mask = torch.tensor(
      self.mask_id, device=input_ids.device, dtype=input_ids.dtype)
    return torch.where(move_mask, mask, input_ids)

  @torch.no_grad()
  def forward(self, input_ids: torch.Tensor, t: torch.Tensor):
    alpha_t = _unsqueeze(self.schedule.alpha_t(t), input_ids)
    p_mask = (1.0 - alpha_t).to(dtype=torch.float32)
    move_mask = (torch.rand_like(input_ids, dtype=torch.float32) < p_mask).to(torch.bool)
    xt = self.corrupt_with_mask(input_ids, move_mask)
    return xt, p_mask

