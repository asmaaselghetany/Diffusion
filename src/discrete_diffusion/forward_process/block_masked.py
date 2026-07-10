"""Per-block absorbing (masked) forward process — Fast-dLLM / BlockGen-absorb."""

from __future__ import annotations

import torch

from .base import ForwardProcess
from .utils import _mask_token_id
from ..noise_schedules.base import NoiseSchedule


class BlockMaskedForwardProcess(ForwardProcess):
  """Mask tokens with prob ``1 - alpha_t``, sharing ``t`` within each block."""

  def __init__(self, tokenizer, schedule: NoiseSchedule, name=None) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.mask_id = _mask_token_id(tokenizer)

  @torch.no_grad()
  def forward(
      self,
      input_ids: torch.Tensor,
      t: torch.Tensor,
      *,
      block_size: int,
  ):
    """Args:
      input_ids: ``[B, L]``
      t: ``[B, L]`` per-position times (constant within each block)
    Returns:
      xt, p_mask (both ``[B, L]``)
    """
    alpha_t = self.schedule.alpha_t(t)
    p_mask = (1.0 - alpha_t).to(dtype=torch.float32)
    move_mask = torch.rand_like(input_ids, dtype=torch.float32) < p_mask
    xt = torch.where(
        move_mask,
        torch.tensor(self.mask_id, device=input_ids.device, dtype=input_ids.dtype),
        input_ids)
    return xt, p_mask


def sample_block_timesteps(
    batch_size: int,
    seq_len: int,
    block_size: int,
    device: torch.device,
    *,
    sampling_eps: float = 1e-3,
    antithetic: bool = False,
) -> torch.Tensor:
  """Sample one ``t`` per block, broadcast to token positions."""
  num_blocks = seq_len // block_size
  eps = torch.rand((batch_size, num_blocks), device=device)
  if antithetic:
    offset = torch.arange(batch_size * num_blocks, device=device)
    offset = (offset / (batch_size * num_blocks)).view(batch_size, num_blocks)
    eps = (eps / (batch_size * num_blocks) + offset) % 1
  t = (1 - sampling_eps) * eps + sampling_eps
  return t.repeat_interleave(block_size, dim=-1)


__all__ = ['BlockMaskedForwardProcess', 'sample_block_timesteps']
