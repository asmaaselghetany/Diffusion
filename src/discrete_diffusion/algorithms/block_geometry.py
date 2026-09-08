"""BlockGen-style block-size mixture / stratification helpers.

Ported from ``third_party/blockgen/algo.py`` ``BlockGenBase.get_block_size``.
Distinct from ``algo.stratified_gamma`` (our *timestep* strat within a block).
"""

from __future__ import annotations

from typing import Sequence

import torch


def parse_block_weights(raw) -> torch.Tensor | None:
  """Parse ``algo.block_weights`` → probability vector over log2 sizes.

  Accepts:
    - ``null`` / empty → None (disabled)
    - space-separated string: ``"0.05 0 0 0 0.95"`` → sizes 1 and 16
    - list/tuple / OmegaConf ListConfig of floats
  Index ``i`` means block size ``2**i``.
  """
  if raw is None:
    return None
  # Hydra CLI lists arrive as ListConfig, not plain list.
  try:
    from omegaconf import ListConfig, OmegaConf
    if isinstance(raw, ListConfig):
      raw = OmegaConf.to_container(raw, resolve=True)
  except ImportError:
    pass
  if isinstance(raw, str):
    raw = raw.strip()
    if not raw:
      return None
    vals = [float(x) for x in raw.split()]
  elif isinstance(raw, (list, tuple)):
    if len(raw) == 0:
      return None
    vals = [float(x) for x in raw]
  else:
    raise TypeError(
        f'algo.block_weights must be null|str|list, got {type(raw)}')
  w = torch.tensor(vals, dtype=torch.float64)
  if w.numel() == 0:
    return None
  s = float(w.sum().item())
  if s <= 0:
    raise ValueError('algo.block_weights must sum to a positive value')
  if abs(s - 1.0) > 1e-6:
    # BlockGen: push remainder onto index 0.
    w = w.clone()
    w[0] = w[0] + (1.0 - s)
  if not torch.isclose(w.sum(), torch.tensor(1.0, dtype=w.dtype), atol=1e-5):
    raise ValueError(f'algo.block_weights must sum to 1, got {w.sum().item()}')
  return w.float()


def sizes_from_weights(weights: torch.Tensor) -> list[int]:
  """Active power-of-two sizes implied by a weight vector (mass > 0)."""
  return [2 ** i for i, p in enumerate(weights.tolist()) if p > 0]


def sample_log_block_size(
    weights: torch.Tensor,
    *,
    mode: str,
    world_size: int,
    global_rank: int,
    accumulate_grad_batches: int,
    current_accumulation_step: int | None,
    seed: int,
    generator: torch.Generator | None,
    u_rv_state: dict,
) -> tuple[int, torch.Generator | None]:
  """Return ``log2(block_size)`` under BlockGen ``block_size_per_gpu`` modes.

  ``u_rv_state`` is a mutable dict holding ``{'u_rv': Tensor|None}`` so
  ``u-stratified`` can reuse one U draw across microbatches of an optimizer
  step (matches BlockGen).
  """
  if weights.numel() == 1:
    # BlockGen shortcut: single weight → always size 1 (log=0).
    return 0, generator

  mode = mode or 'same'
  if mode == 'same':
    log_bs = int(torch.multinomial(weights, 1).item())
    return log_bs, generator

  if mode == 'random':
    if generator is None:
      generator = torch.Generator('cpu')
      generator.manual_seed(int(seed) + int(global_rank))
    log_bs = int(torch.multinomial(weights, 1, generator=generator).item())
    return log_bs, generator

  if mode == 'u-stratified':
    if generator is None:
      generator = torch.Generator('cpu')
      generator.manual_seed(int(seed) + int(global_rank))
    cum = torch.cumsum(weights, dim=0)
    cum = torch.cat([torch.tensor([0.0]), cum], dim=0)
    lo = cum[:-1].reshape(-1, 1)
    hi = cum[1:].reshape(-1, 1)
    n_slots = max(1, int(world_size) * max(1, int(accumulate_grad_batches)))
    if current_accumulation_step is not None:
      if int(current_accumulation_step) == 0 or u_rv_state.get('u_rv') is None:
        u_rv_state['u_rv'] = torch.rand(1, 1)
      u_rv = u_rv_state['u_rv']
    else:
      u_rv = torch.rand(1, 1)
    lin = torch.linspace(0, 1, n_slots + 1)[:-1]
    u_strat = (lin + u_rv) % 1.0
    in_bounds = (u_strat >= lo) & (u_strat < hi)
    idx_per_device = in_bounds.long().argmax(dim=0)
    idx_per_device = idx_per_device[
        torch.randperm(len(idx_per_device), generator=generator)]
    if current_accumulation_step is None:
      slot = int(global_rank) % n_slots
    else:
      slot = int(world_size) * int(current_accumulation_step) + int(global_rank)
      slot = slot % n_slots
    log_bs = int(idx_per_device[slot].item())
    return log_bs, generator

  raise ValueError(
      f'algo.block_size_per_gpu={mode!r} invalid '
      f'(expected same|random|u-stratified)')


def validate_mixture_vs_weights(
    mixture: Sequence[int],
    weights: torch.Tensor | None,
) -> None:
  """Forbid simultaneous list mixture and weighted 2^k mixture."""
  has_mix = bool(mixture)
  has_w = weights is not None
  if has_mix and has_w:
    raise ValueError(
        'algo.block_size_mixture and algo.block_weights are mutually '
        'exclusive — pick one mixture API')


__all__ = [
    'parse_block_weights',
    'sizes_from_weights',
    'sample_log_block_size',
    'validate_mixture_vs_weights',
]
