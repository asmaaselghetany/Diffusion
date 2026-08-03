"""Block mask injection for Qwen2 attention."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from ..block_mask import build_sdpa_mask


@contextmanager
def block_diff_attention_mask(
    model,
    n: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Iterator[None]:
  inner = model.model if hasattr(model, 'model') else model
  additive = build_sdpa_mask(n, block_size, device=device, dtype=dtype)
  original = inner._update_causal_mask

  def _patched(attention_mask, input_tensor, cache_position, past_key_values,
               output_attentions: bool = False):
    del attention_mask, cache_position, past_key_values, output_attentions
    return additive.to(dtype=input_tensor.dtype, device=input_tensor.device)

  inner._update_causal_mask = _patched
  try:
    yield
  finally:
    inner._update_causal_mask = original
