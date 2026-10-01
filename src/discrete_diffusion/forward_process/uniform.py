"""Uniform forward process: random token replacement (legacy UDLM path).

With probability ``(1 - alpha_t)``, replaces each token by a uniformly drawn
token from ``Unif(V \\ {MASK})`` when a mask token is present — matching
BlockGen-with-MASK / hybrid contract. BlockGen / Duo have no MASK and use
plain ``Unif(V)``.
"""

from __future__ import annotations

import torch

from .base import ForwardProcess
from .utils import (
    _effective_vocab_size,
    _mask_token_id,
    resolve_uniform_exclude_ids,
    sample_uniform_excluding_mask,
)
from ..noise_schedules.base import NoiseSchedule


class UniformForwardProcess(ForwardProcess):
  def __init__(self, tokenizer, schedule: NoiseSchedule, name=None) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.vocab_size = _effective_vocab_size(tokenizer)
    try:
      self.mask_id = _mask_token_id(tokenizer)
    except ValueError:
      self.mask_id = None
    self.exclude_ids = resolve_uniform_exclude_ids(
        tokenizer, mask_id=self.mask_id, vocab_size=self.vocab_size)

  @torch.no_grad()
  def forward(self, input_ids: torch.Tensor, t: torch.Tensor):
    alpha_t = self.schedule.alpha_t(t).view(-1, 1)
    p_replace = (1.0 - alpha_t).to(dtype=torch.float32)
    move_mask = (
        torch.rand_like(input_ids, dtype=torch.float32) < p_replace
    ).to(torch.bool)
    uniform_draw = sample_uniform_excluding_mask(
        input_ids.shape,
        vocab_size=self.vocab_size,
        mask_id=self.mask_id,
        device=input_ids.device,
        dtype=input_ids.dtype,
        exclude_ids=self.exclude_ids,
    )
    xt = torch.where(move_mask, uniform_draw, input_ids)
    return xt
