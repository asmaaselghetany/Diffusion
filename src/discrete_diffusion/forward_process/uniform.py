"""Uniform forward process: random token replacement.

With probability `(1 - alpha_t)`, replaces each token by a uniformly drawn
token from the vocabulary (optionally excluding the mask token if present).
"""

from __future__ import annotations

import torch

from .base import ForwardProcess
from .utils import _effective_vocab_size, _unsqueeze
from ..noise_schedules.base import NoiseSchedule


class UniformForwardProcess(ForwardProcess):
  def __init__(self, tokenizer, schedule: NoiseSchedule, name=None) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.vocab_size = _effective_vocab_size(tokenizer)

  @torch.no_grad()
  def corrupt_with_mask(self, input_ids: torch.Tensor, move_mask: torch.Tensor):
    uniform_draw = torch.randint(
      low=0,
      high=self.vocab_size,
      size=input_ids.shape,
      device=input_ids.device,
      dtype=input_ids.dtype,
    )
    return torch.where(move_mask, uniform_draw, input_ids)

  @torch.no_grad()
  def forward(self, input_ids: torch.Tensor, t: torch.Tensor):
    alpha_t = _unsqueeze(self.schedule.alpha_t(t), input_ids)
    p_replace = (1.0 - alpha_t).to(dtype=torch.float32)
    move_mask = (torch.rand_like(input_ids, dtype=torch.float32) < p_replace).to(torch.bool)
    return self.corrupt_with_mask(input_ids, move_mask)
