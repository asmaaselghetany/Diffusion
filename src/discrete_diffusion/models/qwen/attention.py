"""Block mask injection for Qwen2 attention."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from ...contracts.attention_hook import assert_block_attention_hook_compatible
from ..block_mask import (
    build_eval_sdpa_mask,
    build_generation_sdpa_mask,
    build_sdpa_mask,
)


@contextmanager
def block_diff_attention_mask(
    model,
    n: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    padding_mask: torch.Tensor | None = None,
    *,
    intra_block_open: float = 1.0,
    x0_causal: bool = False,
) -> Iterator[None]:
  assert_block_attention_hook_compatible(model)
  inner = model.model if hasattr(model, 'model') else model
  original = inner._update_causal_mask
  open_p = float(intra_block_open)
  x0_c = bool(x0_causal)

  def _patched(attention_mask, input_tensor, cache_position, past_key_values,
               output_attentions: bool = False):
    del cache_position, past_key_values, output_attentions
    # Prefer explicit padding_mask from the trainer; fall back to HF mask.
    pad = padding_mask if padding_mask is not None else attention_mask
    out = build_sdpa_mask(
        n, block_size, device=input_tensor.device, dtype=input_tensor.dtype,
        padding_mask=pad, intra_block_open=open_p, x0_causal=x0_c)
    return out.to(dtype=input_tensor.dtype, device=input_tensor.device)

  inner._update_causal_mask = _patched
  try:
    yield
  finally:
    inner._update_causal_mask = original


@contextmanager
def eval_block_causal_attention_mask(
    model,
    seq_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    cache_seq_len: int = 0,
    padding_mask: torch.Tensor | None = None,
) -> Iterator[None]:
  """Inject Hub ``eval_block_diff_mask`` (single-stream block-causal)."""
  assert_block_attention_hook_compatible(model)
  inner = model.model if hasattr(model, 'model') else model
  original = inner._update_causal_mask

  def _patched(attention_mask, input_tensor, cache_position, past_key_values,
               output_attentions: bool = False):
    del cache_position, past_key_values, output_attentions
    pad = padding_mask if padding_mask is not None else attention_mask
    out = build_eval_sdpa_mask(
        seq_len, block_size, device=input_tensor.device,
        dtype=input_tensor.dtype, cache_seq_len=cache_seq_len,
        padding_mask=pad)
    return out.to(dtype=input_tensor.dtype, device=input_tensor.device)

  inner._update_causal_mask = _patched
  try:
    yield
  finally:
    inner._update_causal_mask = original


@contextmanager
def block_generation_attention_mask(
    model,
    ctx_len: int,
    xt_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    padding_mask: torch.Tensor | None = None,
    *,
    x0_causal: bool = False,
) -> Iterator[None]:
  """Inject BlockGen ``block_generation_mask`` for ``[prefix | xt_block]``."""
  assert_block_attention_hook_compatible(model)
  inner = model.model if hasattr(model, 'model') else model
  original = inner._update_causal_mask

  def _patched(attention_mask, input_tensor, cache_position, past_key_values,
               output_attentions: bool = False):
    del cache_position, past_key_values, output_attentions
    pad = padding_mask if padding_mask is not None else attention_mask
    out = build_generation_sdpa_mask(
        ctx_len, xt_len, block_size,
        device=input_tensor.device, dtype=input_tensor.dtype,
        padding_mask=pad, x0_causal=x0_causal)
    return out.to(dtype=input_tensor.dtype, device=input_tensor.device)

  inner._update_causal_mask = _patched
  try:
    yield
  finally:
    inner._update_causal_mask = original
