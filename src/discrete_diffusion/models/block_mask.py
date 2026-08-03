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
) -> torch.Tensor:
  allowed = build_block_diff_bool_mask(n, block_size, device)
  mask = torch.zeros((1, 1, n * 2, n * 2), device=device, dtype=dtype)
  return mask.masked_fill(~allowed, float('-inf'))


def build_causal_bool_mask(seq_len: int, device: torch.device | str = 'cpu') -> torch.Tensor:
  q_idx = torch.arange(seq_len, device=device)[:, None]
  kv_idx = torch.arange(seq_len, device=device)[None, :]
  return kv_idx <= q_idx


def build_flex_block_mask(n: int, block_size: int):
  from torch.nn.attention.flex_attention import create_block_mask
  return create_block_mask(
      partial(block_diff_mask, block_size=block_size, n=n),
      B=None, H=None, Q_LEN=n * 2, KV_LEN=n * 2)
