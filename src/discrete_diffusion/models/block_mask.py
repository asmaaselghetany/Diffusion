"""Block-diffusion attention masks."""

from __future__ import annotations

from functools import partial

import torch


def block_diff_mask(
    b,
    h,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    block_size: int,
    n: int,
) -> torch.Tensor:
  """Boolean allow-mask for concat(xt, x0) with length ``2n``.

  Three components (block diffusion literature):
  - block-diagonal within xt and within x0
  - offset block-causal: xt attends to previous x0 blocks
  - block-causal among x0 tokens
  """
  del b, h
  x0_flag_q = q_idx >= n
  x0_flag_kv = kv_idx >= n

  block_q = torch.where(
      x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
  block_kv = torch.where(
      x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
  offset_block_causal = (
      (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0))
  block_causal = (
      (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1))
  return block_diagonal | offset_block_causal | block_causal


def build_block_diff_bool_mask(
    n: int, block_size: int, device: torch.device | str = 'cpu') -> torch.Tensor:
  q_idx = torch.arange(n * 2, device=device)[:, None]
  kv_idx = torch.arange(n * 2, device=device)[None, :]
  return block_diff_mask(
      None, None, q_idx, kv_idx, block_size=block_size, n=n)


def build_sdpa_mask(
    n: int,
    block_size: int,
    device: torch.device | str = 'cpu',
    dtype: torch.dtype = torch.float32,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  allowed = build_block_diff_bool_mask(n, block_size, device)
  mask = torch.zeros((1, 1, n * 2, n * 2), device=device, dtype=dtype)
  mask = mask.masked_fill(~allowed, float('-inf'))
  return apply_padding_to_sdpa_mask(mask, padding_mask, n)


def apply_padding_to_sdpa_mask(
    sdpa_mask: torch.Tensor,
    padding_mask: torch.Tensor | None,
    n: int,
) -> torch.Tensor:
  """Block attention to/from padded positions on the ``2n`` concat grid."""
  if padding_mask is None:
    return sdpa_mask
  finfo_min = torch.finfo(sdpa_mask.dtype).min
  pad = padding_mask
  if pad.dtype != torch.bool:
    pad = pad != 0
  if pad.shape[-1] == n:
    pad = torch.cat([pad, pad], dim=-1)
  elif pad.shape[-1] != 2 * n:
    raise ValueError(
        f'padding_mask last dim {pad.shape[-1]} must equal n={n} or 2n={2 * n}')
  batch = pad.shape[0]
  if sdpa_mask.shape[0] == 1 and batch > 1:
    base = sdpa_mask.expand(batch, -1, -1, -1).clone()
  elif sdpa_mask.shape[0] == batch:
    base = sdpa_mask
  else:
    base = sdpa_mask.expand(batch, -1, -1, -1).clone()
  bad_q = (~pad).unsqueeze(1).unsqueeze(-1)
  bad_k = (~pad).unsqueeze(1).unsqueeze(-2)
  return base.masked_fill(bad_q, finfo_min).masked_fill(bad_k, finfo_min)


def build_causal_bool_mask(seq_len: int, device: torch.device | str = 'cpu') -> torch.Tensor:
  q_idx = torch.arange(seq_len, device=device)[:, None]
  kv_idx = torch.arange(seq_len, device=device)[None, :]
  return kv_idx <= q_idx


def build_flex_block_mask(n: int, block_size: int):
  from torch.nn.attention.flex_attention import create_block_mask
  return create_block_mask(
      partial(block_diff_mask, block_size=block_size, n=n),
      B=None, H=None, Q_LEN=n * 2, KV_LEN=n * 2)
