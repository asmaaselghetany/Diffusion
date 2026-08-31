"""Per-block absorbing (masked) forward process."""

from __future__ import annotations

import torch

from .base import ForwardProcess
from .utils import _mask_token_id
from ..noise_schedules.base import NoiseSchedule


class BlockMaskedForwardProcess(ForwardProcess):
  """Mask tokens with prob ``1 - alpha_t``, sharing ``t`` within each block.

  Token-wise Bernoulli masking is already *partial within a block* (Fast-dLLM
  style). Complementary views (``m`` and ``~m``) are **not** applied here —
  they are batch-doubled in ``BlockTrainer.nll`` to match Fast-dLLM v2.
  """

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
      complementary: bool = False,
      return_move_mask: bool = False,
      mask_schedule: str = 'alpha',
  ):
    """Args:
      input_ids: ``[B, L]``
      t: ``[B, L]`` per-position times (constant within each block)
      complementary: **ignored** (kept for call-site compat; must be False).
        Paired ``m`` / ``~m`` views are built in the trainer.
      return_move_mask: if True, also return the Bernoulli mask ``[B, L]``
      mask_schedule:
        - ``alpha`` (default): ``p_mask = 1 - alpha_t(t)`` (LogLinear)
        - ``fast_dllm``: ``p_mask = (1-eps)*t + eps`` (Fast-dLLM v2 modeling.py)
    Returns:
      xt, p_mask  — or xt, p_mask, move_mask when ``return_move_mask``.
    """
    del block_size  # geometry encoded in ``t`` (constant per block)
    if complementary:
      raise ValueError(
          'BlockMaskedForwardProcess.complementary is retired. '
          'Enable algo.complementary_masks on BlockTrainer (paired m/~m '
          'batch doubling, Fast-dLLM v2).')
    if mask_schedule == 'fast_dllm':
      eps = float(getattr(self.schedule, 'eps', 1e-3))
      p_mask = ((1.0 - eps) * t + eps).to(dtype=torch.float32)
    elif mask_schedule == 'alpha':
      alpha_t = self.schedule.alpha_t(t)
      p_mask = (1.0 - alpha_t).to(dtype=torch.float32)
    else:
      raise ValueError(
          f'Unknown mask_schedule={mask_schedule!r} '
          f'(expected alpha|fast_dllm)')
    move_mask = torch.rand_like(input_ids, dtype=torch.float32) < p_mask
    xt = torch.where(
        move_mask,
        torch.tensor(self.mask_id, device=input_ids.device, dtype=input_ids.dtype),
        input_ids)
    if return_move_mask:
      return xt, p_mask, move_mask
    return xt, p_mask


def complementary_pair_from_mask(
    input_ids: torch.Tensor,
    move_mask: torch.Tensor,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Build Fast-dLLM complementary views from one Bernoulli mask.

  View A uses ``m``; view B uses exact ``~m``. Same clean ``input_ids``.
  """
  mid = torch.tensor(mask_id, device=input_ids.device, dtype=input_ids.dtype)
  xt_a = torch.where(move_mask, mid, input_ids)
  xt_b = torch.where(~move_mask, mid, input_ids)
  return xt_a, xt_b


def sample_block_timesteps(
    batch_size: int,
    seq_len: int,
    block_size: int,
    device: torch.device,
    *,
    sampling_eps: float = 1e-3,
    antithetic: bool = False,
    stratified_gamma: float | None = None,
) -> torch.Tensor:
  """Sample one ``t`` per block, broadcast to token positions."""
  num_blocks = seq_len // block_size
  if stratified_gamma is not None:
    grid = torch.linspace(0, 1, num_blocks + 1, device=device)[:-1]
    offset = torch.rand((batch_size, 1), device=device)
    strat = (grid.unsqueeze(0) + offset) % 1.0
    rand = torch.rand((batch_size, num_blocks), device=device)
    gamma = float(stratified_gamma)
    eps = gamma * strat + (1.0 - gamma) * rand
  else:
    eps = torch.rand((batch_size, num_blocks), device=device)
  if antithetic:
    offset = torch.arange(batch_size * num_blocks, device=device, dtype=torch.float32)
    offset = (offset / (batch_size * num_blocks)).view(batch_size, num_blocks)
    # Permute block columns so antithetic stratification is not tied to position.
    perm = torch.argsort(torch.rand(batch_size, num_blocks, device=device), dim=-1)
    offset = torch.gather(offset, 1, perm)
    eps = (eps / (batch_size * num_blocks) + offset) % 1
  t = (1 - sampling_eps) * eps + sampling_eps
  return t.repeat_interleave(block_size, dim=-1)


__all__ = [
    'BlockMaskedForwardProcess',
    'complementary_pair_from_mask',
    'sample_block_timesteps',
]
