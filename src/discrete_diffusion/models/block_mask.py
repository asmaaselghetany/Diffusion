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
    intra_block_open: float = 1.0,
    x0_causal: bool = False,
) -> torch.Tensor:
  """Boolean allow-mask for concat(xt, x0) with length ``2n``.

  Three components (block diffusion literature):
  - block-diagonal within xt and within x0
  - offset block-causal: xt attends to previous x0 blocks
  - among x0 tokens: token-causal if ``x0_causal`` else block-causal
    (BlockGen ``model.x0_causal`` for multi-size AR↔diff KV share)

  ``intra_block_open`` ∈ [0, 1] (Unifusion-style anneal, block-local):
  - 0 → causal within each block (kv ≤ q on the same stream)
  - 1 → full bidirectional within each block (default / legacy)
  Across-block structure is unchanged.
  """
  del b, h
  x0_flag_q = q_idx >= n
  x0_flag_kv = kv_idx >= n

  block_q = torch.where(
      x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
  block_kv = torch.where(
      x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
  # Token indices on the shared 0..n-1 RoPE grid (both streams).
  pos_q = torch.where(x0_flag_q, q_idx - n, q_idx)
  pos_kv = torch.where(x0_flag_kv, kv_idx - n, kv_idx)
  open_p = float(max(0.0, min(1.0, intra_block_open)))
  if open_p >= 1.0 - 1e-8:
    intra_ok = torch.ones_like(block_diagonal)
  elif open_p <= 1e-8:
    intra_ok = pos_kv <= pos_q
  else:
    # Gradually open look-ahead inside the block: ahead = open_p * block_size.
    ahead = open_p * float(block_size)
    intra_ok = pos_kv.to(torch.float32) <= (
        pos_q.to(torch.float32) + ahead)
  block_diagonal = block_diagonal & intra_ok
  if x0_causal:
    # BlockGen: x0↔x0 is only token-causal (no bi diagonal on clean half).
    block_diagonal = block_diagonal & ~(x0_flag_q & x0_flag_kv)

  offset_block_causal = (
      (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0))
  if x0_causal:
    # Token-causal on the clean half (BlockGen multi-block + ARPC).
    x0_to_x0 = (q_idx >= kv_idx) & x0_flag_q & x0_flag_kv
  else:
    x0_to_x0 = (
        (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1))
  return block_diagonal | offset_block_causal | x0_to_x0


def build_block_diff_bool_mask(
    n: int,
    block_size: int,
    device: torch.device | str = 'cpu',
    *,
    intra_block_open: float = 1.0,
    x0_causal: bool = False,
) -> torch.Tensor:
  q_idx = torch.arange(n * 2, device=device)[:, None]
  kv_idx = torch.arange(n * 2, device=device)[None, :]
  return block_diff_mask(
      None, None, q_idx, kv_idx, block_size=block_size, n=n,
      intra_block_open=intra_block_open, x0_causal=x0_causal)


def build_sdpa_mask(
    n: int,
    block_size: int,
    device: torch.device | str = 'cpu',
    dtype: torch.dtype = torch.float32,
    padding_mask: torch.Tensor | None = None,
    *,
    intra_block_open: float = 1.0,
    x0_causal: bool = False,
) -> torch.Tensor:
  allowed = build_block_diff_bool_mask(
      n, block_size, device, intra_block_open=intra_block_open,
      x0_causal=x0_causal)
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


def eval_block_diff_mask(
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
  """Hub ``eval_block_diff_mask``: single-stream block-causal allow-mask.

  Tokens attend to all positions in the same or earlier blocks (full within
  the current block). Used at decode time when ``xt == x0`` for committed
  prefix — mathematically equivalent to dual-stream train mask on that state.
  """
  return (q_idx // block_size) >= (kv_idx // block_size)


def build_eval_block_bool_mask(
    seq_len: int,
    block_size: int,
    device: torch.device | str = 'cpu',
    *,
    cache_seq_len: int = 0,
) -> torch.Tensor:
  """Boolean allow-mask for single-stream decode of length ``seq_len``.

  ``cache_seq_len`` shifts query indices when K/V for a prefix already live
  in ``past_key_values`` (Hub ``eval_mask``).
  """
  q_idx = torch.arange(seq_len, device=device)[:, None] + cache_seq_len
  kv_idx = torch.arange(seq_len + cache_seq_len, device=device)[None, :]
  return eval_block_diff_mask(q_idx, kv_idx, block_size=block_size)


def apply_padding_to_eval_sdpa_mask(
    sdpa_mask: torch.Tensor,
    padding_mask: torch.Tensor | None,
    seq_len: int,
    *,
    cache_seq_len: int = 0,
) -> torch.Tensor:
  """Block attention to/from padded positions on the eval ``[q, kv]`` grid."""
  if padding_mask is None:
    return sdpa_mask
  finfo_min = torch.finfo(sdpa_mask.dtype).min
  pad = padding_mask
  if pad.dtype != torch.bool:
    pad = pad != 0
  kv_len = seq_len + cache_seq_len
  if pad.shape[-1] == kv_len:
    pad_k = pad
    pad_q = pad[:, cache_seq_len:cache_seq_len + seq_len]
  elif pad.shape[-1] == seq_len and cache_seq_len == 0:
    pad_q = pad_k = pad
  else:
    raise ValueError(
        f'padding_mask last dim {pad.shape[-1]} must equal seq_len={seq_len} '
        f'(cache_seq_len=0) or kv_len={kv_len}')
  batch = pad.shape[0]
  if sdpa_mask.shape[0] == 1 and batch > 1:
    base = sdpa_mask.expand(batch, -1, -1, -1).clone()
  elif sdpa_mask.shape[0] == batch:
    base = sdpa_mask
  else:
    base = sdpa_mask.expand(batch, -1, -1, -1).clone()
  bad_q = (~pad_q).unsqueeze(1).unsqueeze(-1)
  bad_k = (~pad_k).unsqueeze(1).unsqueeze(-2)
  return base.masked_fill(bad_q, finfo_min).masked_fill(bad_k, finfo_min)


def build_eval_sdpa_mask(
    seq_len: int,
    block_size: int,
    device: torch.device | str = 'cpu',
    dtype: torch.dtype = torch.float32,
    *,
    cache_seq_len: int = 0,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  allowed = build_eval_block_bool_mask(
      seq_len, block_size, device, cache_seq_len=cache_seq_len)
  q = seq_len
  k = seq_len + cache_seq_len
  mask = torch.zeros((1, 1, q, k), device=device, dtype=dtype)
  mask = mask.masked_fill(~allowed, float('-inf'))
  return apply_padding_to_eval_sdpa_mask(
      mask, padding_mask, seq_len, cache_seq_len=cache_seq_len)


def build_flex_block_mask(n: int, block_size: int):
  from torch.nn.attention.flex_attention import create_block_mask
  return create_block_mask(
      partial(block_diff_mask, block_size=block_size, n=n),
      B=None, H=None, Q_LEN=n * 2, KV_LEN=n * 2)


def block_generation_mask(
    b,
    h,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    ctx_len: int,
    xt_len: int,
    block_size: int,
    x0_causal: bool = False,
) -> torch.Tensor:
  """BlockGen generate allow-mask: layout ``[x0_prefix | xt_block]``.

  Port of ``third_party/blockgen/models/block_dit.py::block_generation_mask``.

  - xt → x0: all allowed (condition on clean prefix)
  - xt → xt: block-diagonal
  - x0 → x0: token-causal if ``x0_causal`` else block-causal
  - x0 → xt: forbidden
  """
  del b, h, xt_len
  x0_q = q_idx < ctx_len
  x0_kv = kv_idx < ctx_len
  xt_to_x0 = (~x0_q) & x0_kv
  xt_to_xt = (
      (((q_idx - ctx_len) // block_size)
       == ((kv_idx - ctx_len) // block_size))
      & (~x0_q) & (~x0_kv))
  if x0_causal:
    x0_to_x0 = (q_idx >= kv_idx) & x0_q & x0_kv
  else:
    x0_to_x0 = (
        (q_idx // block_size >= kv_idx // block_size) & x0_q & x0_kv)
  return xt_to_x0 | xt_to_xt | x0_to_x0


def build_block_generation_bool_mask(
    ctx_len: int,
    xt_len: int,
    block_size: int,
    device: torch.device | str = 'cpu',
    *,
    x0_causal: bool = False,
) -> torch.Tensor:
  """Boolean allow-mask for packed length ``ctx_len + xt_len``."""
  total = int(ctx_len) + int(xt_len)
  q_idx = torch.arange(total, device=device)[:, None]
  kv_idx = torch.arange(total, device=device)[None, :]
  return block_generation_mask(
      None, None, q_idx, kv_idx,
      ctx_len=int(ctx_len), xt_len=int(xt_len),
      block_size=int(block_size), x0_causal=x0_causal)


def build_generation_sdpa_mask(
    ctx_len: int,
    xt_len: int,
    block_size: int,
    device: torch.device | str = 'cpu',
    dtype: torch.dtype = torch.float32,
    padding_mask: torch.Tensor | None = None,
    *,
    x0_causal: bool = False,
) -> torch.Tensor:
  """Additive SDPA mask for BlockGen ``[prefix | block]`` generate packing."""
  total = int(ctx_len) + int(xt_len)
  allowed = build_block_generation_bool_mask(
      ctx_len, xt_len, block_size, device, x0_causal=x0_causal)
  mask = torch.zeros((1, 1, total, total), device=device, dtype=dtype)
  mask = mask.masked_fill(~allowed, float('-inf'))
  if padding_mask is None:
    return mask
  # Padding over the packed sequence (prefix then block).
  finfo_min = torch.finfo(mask.dtype).min
  pad = padding_mask
  if pad.dtype != torch.bool:
    pad = pad != 0
  if pad.shape[-1] != total:
    raise ValueError(
        f'generation padding_mask last dim {pad.shape[-1]} != packed {total}')
  batch = pad.shape[0]
  if mask.shape[0] == 1 and batch > 1:
    base = mask.expand(batch, -1, -1, -1).clone()
  elif mask.shape[0] == batch:
    base = mask
  else:
    base = mask.expand(batch, -1, -1, -1).clone()
  bad_q = (~pad).unsqueeze(1).unsqueeze(-1)
  bad_k = (~pad).unsqueeze(1).unsqueeze(-2)
  return base.masked_fill(bad_q, finfo_min).masked_fill(bad_k, finfo_min)

