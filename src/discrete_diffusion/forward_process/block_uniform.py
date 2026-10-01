"""Per-block uniform forward process — BlockGen-uniform + MASK-safe Qwen port.

BlockGen / Duo ``UniformState`` never append a MASK token, so their draw is
plain ``Unif(V)``. Our conversion stack always keeps MASK for the shared
masked arm; the matching contract is therefore ``Unif(V \\ {MASK})`` (same as
``BlockHybridForwardProcess``'s uniform branch).

Ablation: ``simplex_mode='blockgen'`` restores literal ``Unif(V)`` even when
MASK exists as a vocab id (conversion BlockGen-simplex cell).
"""

from __future__ import annotations

import torch

from .base import ForwardProcess
from .utils import (
    _effective_vocab_size,
    _mask_token_id,
    normalize_uniform_simplex_mode,
    resolve_uniform_exclude_ids,
    sample_uniform_excluding_mask,
)
from ..noise_schedules.base import NoiseSchedule


class BlockUniformForwardProcess(ForwardProcess):
  """Replace tokens with uniform vocab draw, sharing ``t`` within each block."""

  def __init__(
      self,
      tokenizer,
      schedule: NoiseSchedule,
      name=None,
      *,
      simplex_mode: str = 'conversion',
  ) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.vocab_size = _effective_vocab_size(tokenizer)
    try:
      self.mask_id = _mask_token_id(tokenizer)
    except ValueError:
      self.mask_id = None
    self.simplex_mode = normalize_uniform_simplex_mode(simplex_mode)
    self.exclude_ids = resolve_uniform_exclude_ids(
        tokenizer, mask_id=self.mask_id, vocab_size=self.vocab_size,
        mode=self.simplex_mode)

  @torch.no_grad()
  def forward(
      self,
      input_ids: torch.Tensor,
      t: torch.Tensor,
      *,
      block_size: int,
      return_move_mask: bool = False,
  ):
    del block_size
    alpha_t = self.schedule.alpha_t(t)
    p_replace = (1.0 - alpha_t).to(dtype=torch.float32)
    move_mask = torch.rand_like(input_ids, dtype=torch.float32) < p_replace
    # BlockGen ablation: mask_id=None so empty exclude → full V.
    mid = None if self.simplex_mode == 'blockgen' else self.mask_id
    uniform_draw = sample_uniform_excluding_mask(
        input_ids.shape,
        vocab_size=self.vocab_size,
        mask_id=mid,
        device=input_ids.device,
        dtype=input_ids.dtype,
        exclude_ids=self.exclude_ids,
    )
    xt = torch.where(move_mask, uniform_draw, input_ids)
    if return_move_mask:
      return xt, move_mask
    return xt


__all__ = ['BlockUniformForwardProcess']
