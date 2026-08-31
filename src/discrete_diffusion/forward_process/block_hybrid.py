"""Per-block hybrid (mask + uniform) forward process — B4 / GIDD-inspired."""

from __future__ import annotations

import torch

from .base import ForwardProcess
from .utils import _effective_vocab_size, _mask_token_id
from ..noise_schedules.base import NoiseSchedule


class BlockHybridForwardProcess(ForwardProcess):
  """Corrupt with mask+uniform mix, sharing ``t`` within each block.

  For each token, with probability ``p_replace = 1 - alpha_t(t)``:
    - with probability ``1 - p_uniform`` → MASK
    - with probability ``p_uniform`` → Unif(V \\ {MASK})

  ``p_uniform=0`` recovers absorbing; ``p_uniform=1`` recovers uniform-like
  replacement (excluding MASK as the absorbing atom).
  """

  def __init__(
      self,
      tokenizer,
      schedule: NoiseSchedule,
      name=None,
      *,
      p_uniform: float = 0.1,
  ) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.mask_id = _mask_token_id(tokenizer)
    self.vocab_size = _effective_vocab_size(tokenizer)
    self.p_uniform = float(p_uniform)
    if not (0.0 <= self.p_uniform <= 1.0):
      raise ValueError(
          f'hybrid p_uniform must be in [0, 1], got {self.p_uniform}')

  def _uniform_excluding_mask(self, shape, device, dtype) -> torch.Tensor:
    """Sample Unif({0..V-1} \\ {mask_id})."""
    u = torch.randint(
        0, self.vocab_size - 1, shape, device=device, dtype=dtype)
    return torch.where(u >= self.mask_id, u + 1, u)

  @torch.no_grad()
  def forward(
      self,
      input_ids: torch.Tensor,
      t: torch.Tensor,
      *,
      block_size: int,
  ):
    del block_size
    alpha_t = self.schedule.alpha_t(t)
    p_replace = (1.0 - alpha_t).to(dtype=torch.float32)
    move_mask = torch.rand_like(input_ids, dtype=torch.float32) < p_replace
    use_uniform = (
        torch.rand_like(input_ids, dtype=torch.float32) < self.p_uniform)
    mid = torch.tensor(
        self.mask_id, device=input_ids.device, dtype=input_ids.dtype)
    unif = self._uniform_excluding_mask(
        input_ids.shape, input_ids.device, input_ids.dtype)
    replacement = torch.where(use_uniform, unif, mid)
    return torch.where(move_mask, replacement, input_ids)


__all__ = ['BlockHybridForwardProcess']
