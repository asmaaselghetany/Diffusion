"""Fast-dLLM-style DualCache (replace_position splice) for Qwen block_diff decode.

Decode-only. Training still uses the stock HF forward + ``block_diff_attention_mask``.

Prefill stores per-layer K/V for the truncated dual stream ``concat(xt[:A], x0[:A])``.
Replace recomputes only a window ``[w0:w1)`` on both halves, splicing into the
cache (same idea as Fast-dLLM ``use_block_cache`` + ``replace_position``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..block_mask import build_sdpa_mask


@dataclass
class DualCache:
  """Per-layer K/V for a dual-stream sequence of length ``2 * active_len``."""

  active_len: int
  layers: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

  def clear(self) -> None:
    self.layers.clear()


def replace_position_indices(
    active_len: int, w0: int, w1: int, device: torch.device,
) -> torch.Tensor:
  """Token indices to splice: xt[w0:w1] and x0[w0:w1] in concat layout."""
  if not (0 <= w0 < w1 <= active_len):
    raise ValueError(f'bad window [{w0},{w1}) for active_len={active_len}')
  xt_idx = torch.arange(w0, w1, device=device)
  x0_idx = torch.arange(active_len + w0, active_len + w1, device=device)
  return torch.cat([xt_idx, x0_idx], dim=0)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  if n_rep == 1:
    return hidden_states
  b, n_kv, s, d = hidden_states.shape
  return (
      hidden_states[:, :, None, :, :]
      .expand(b, n_kv, n_rep, s, d)
      .reshape(b, n_kv * n_rep, s, d)
  )


def _apply_rope(q, k, cos, sin):
  """Apply RoPE; cos/sin shaped for the query/key sequence length."""
  from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
  return apply_rotary_pos_emb(q, k, cos, sin)


def _sdpa_mask_for_queries(
    full_mask: torch.Tensor, query_idx: torch.Tensor,
) -> torch.Tensor:
  """Slice full ``[1,1,2A,2A]`` mask down to query rows ``query_idx``."""
  # full_mask: [1, 1, 2A, 2A]
  return full_mask.index_select(2, query_idx)


def _project_qkv(attn, hidden: torch.Tensor):
  bsz, q_len, _ = hidden.shape
  q = attn.q_proj(hidden).view(
      bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
  k = attn.k_proj(hidden).view(
      bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
  v = attn.v_proj(hidden).view(
      bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
  return q, k, v


@torch.no_grad()
def dual_stream_prefill(
    hf_lm,
    input_ids: torch.Tensor,
    *,
    active_len: int,
    block_size: int,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, DualCache]:
  """Full truncated dual-stream forward; returns xt-half logits + cache."""
  if input_ids.shape[1] != 2 * active_len:
    raise ValueError(
        f'prefill expected seq {2 * active_len}, got {input_ids.shape[1]}')
  inner = hf_lm.model
  device = input_ids.device
  dtype = next(hf_lm.parameters()).dtype
  full_mask = build_sdpa_mask(
      active_len, block_size, device=device, dtype=dtype,
      padding_mask=attention_mask)

  hidden = inner.embed_tokens(input_ids)
  # rotary_emb expects hidden for dtype/device; returns (cos, sin) over positions
  position_embeddings = inner.rotary_emb(hidden, position_ids)

  cache = DualCache(active_len=active_len)
  for layer in inner.layers:
    residual = hidden
    h = layer.input_layernorm(hidden)
    attn = layer.self_attn
    q, k, v = _project_qkv(attn, h)
    cos, sin = position_embeddings
    q, k = _apply_rope(q, k, cos, sin)
    cache.layers.append((k.contiguous(), v.contiguous()))
    k_rep = _repeat_kv(k, attn.num_key_value_groups)
    v_rep = _repeat_kv(v, attn.num_key_value_groups)
    out = F.scaled_dot_product_attention(
        q, k_rep, v_rep, attn_mask=full_mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(h.shape[0], h.shape[1], -1)
    out = attn.o_proj(out)
    hidden = residual + out
    residual = hidden
    h = layer.post_attention_layernorm(hidden)
    hidden = residual + layer.mlp(h)

  hidden = inner.norm(hidden)
  logits = hf_lm.lm_head(hidden)
  return logits[:, :active_len, :], cache


@torch.no_grad()
def dual_stream_replace(
    hf_lm,
    xt: torch.Tensor,
    x0: torch.Tensor,
    *,
    active_len: int,
    window: tuple[int, int],
    block_size: int,
    cache: DualCache,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Recompute window tokens; splice K/V; return full xt-half logits buffer.

  Prefix logits outside the window are left as zeros (sampler only writes the
  active window). Window logits are returned in the correct absolute columns.
  """
  w0, w1 = window
  if cache.active_len != active_len:
    raise ValueError(
        f'cache active_len {cache.active_len} != {active_len}')
  if not cache.layers:
    raise ValueError('DualCache is empty; call prefill first')

  device = xt.device
  dtype = next(hf_lm.parameters()).dtype
  bsz = xt.shape[0]
  idx = replace_position_indices(active_len, w0, w1, device)
  win_ids = torch.cat([xt[:, w0:w1], x0[:, w0:w1]], dim=-1)
  # Shared RoPE ids for the window (same scheme as full dual stream).
  pos_win = torch.arange(w0, w1, device=device)
  position_ids = pos_win.repeat(2).unsqueeze(0).expand(bsz, -1).contiguous()

  inner = hf_lm.model
  full_mask = build_sdpa_mask(
      active_len, block_size, device=device, dtype=dtype,
      padding_mask=attention_mask)
  q_mask = _sdpa_mask_for_queries(full_mask, idx)

  hidden = inner.embed_tokens(win_ids)
  # Build full-length cos/sin then index — RoPE must match absolute positions.
  # Dummy full hidden for rotary_emb length; it only needs seq dim.
  dummy = torch.zeros(
      bsz, 2 * active_len, hidden.shape[-1], device=device, dtype=hidden.dtype)
  full_pos = torch.arange(active_len, device=device).repeat(2).unsqueeze(0).expand(
      bsz, -1).contiguous()
  cos_full, sin_full = inner.rotary_emb(dummy, full_pos)
  # cos_full: [B, 2A, D] or [B, 1, 2A, D] depending on transformers version
  cos_w = cos_full.index_select(-2 if cos_full.dim() == 4 else 1, idx)
  sin_w = sin_full.index_select(-2 if sin_full.dim() == 4 else 1, idx)

  for li, layer in enumerate(inner.layers):
    residual = hidden
    h = layer.input_layernorm(hidden)
    attn = layer.self_attn
    q, k, v = _project_qkv(attn, h)
    q, k = _apply_rope(q, k, cos_w, sin_w)
    # Splice into cache (pre-repeat_kv layout).
    k_cache, v_cache = cache.layers[li]
    k_cache = k_cache.clone()
    v_cache = v_cache.clone()
    k_cache[:, :, idx, :] = k
    v_cache[:, :, idx, :] = v
    cache.layers[li] = (k_cache, v_cache)

    k_rep = _repeat_kv(k_cache, attn.num_key_value_groups)
    v_rep = _repeat_kv(v_cache, attn.num_key_value_groups)
    out = F.scaled_dot_product_attention(
        q, k_rep, v_rep, attn_mask=q_mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(h.shape[0], h.shape[1], -1)
    out = attn.o_proj(out)
    hidden = residual + out
    residual = hidden
    h = layer.post_attention_layernorm(hidden)
    hidden = residual + layer.mlp(h)

  hidden = inner.norm(hidden)
  win_logits = hf_lm.lm_head(hidden)  # [B, 2W, V]
  wlen = w1 - w0
  xt_win = win_logits[:, :wlen, :]
  out = torch.zeros(
      bsz, active_len, xt_win.shape[-1], device=device, dtype=xt_win.dtype)
  out[:, w0:w1, :] = xt_win
  return out


@torch.no_grad()
def single_stream_prefill(
    hf_lm,
    input_ids: torch.Tensor,
    *,
    active_len: int,
    block_size: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, DualCache]:
  """Hub-shaped DualCache prefill on single-stream ``xt[:A]`` + eval mask."""
  from ..block_mask import build_eval_sdpa_mask

  if input_ids.shape[1] != active_len:
    raise ValueError(
        f'prefill expected seq {active_len}, got {input_ids.shape[1]}')
  inner = hf_lm.model
  device = input_ids.device
  dtype = next(hf_lm.parameters()).dtype
  full_mask = build_eval_sdpa_mask(
      active_len, block_size, device=device, dtype=dtype,
      padding_mask=attention_mask)

  hidden = inner.embed_tokens(input_ids)
  position_ids = torch.arange(
      active_len, device=device).unsqueeze(0).expand(
          input_ids.shape[0], active_len).contiguous()
  position_embeddings = inner.rotary_emb(hidden, position_ids)

  cache = DualCache(active_len=active_len)
  for layer in inner.layers:
    residual = hidden
    h = layer.input_layernorm(hidden)
    attn = layer.self_attn
    q, k, v = _project_qkv(attn, h)
    cos, sin = position_embeddings
    q, k = _apply_rope(q, k, cos, sin)
    cache.layers.append((k.contiguous(), v.contiguous()))
    k_rep = _repeat_kv(k, attn.num_key_value_groups)
    v_rep = _repeat_kv(v, attn.num_key_value_groups)
    out = F.scaled_dot_product_attention(
        q, k_rep, v_rep, attn_mask=full_mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(h.shape[0], h.shape[1], -1)
    out = attn.o_proj(out)
    hidden = residual + out
    residual = hidden
    h = layer.post_attention_layernorm(hidden)
    hidden = residual + layer.mlp(h)

  hidden = inner.norm(hidden)
  logits = hf_lm.lm_head(hidden)
  return logits, cache


@torch.no_grad()
def single_stream_replace(
    hf_lm,
    xt: torch.Tensor,
    *,
    active_len: int,
    window: tuple[int, int],
    block_size: int,
    cache: DualCache,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Hub ``replace_position`` splice on single-stream DualCache."""
  from ..block_mask import build_eval_sdpa_mask

  w0, w1 = window
  if cache.active_len != active_len:
    raise ValueError(
        f'cache active_len {cache.active_len} != {active_len}')
  if not cache.layers:
    raise ValueError('DualCache is empty; call prefill first')
  if not (0 <= w0 < w1 <= active_len):
    raise ValueError(f'bad window [{w0},{w1}) for active_len={active_len}')

  device = xt.device
  dtype = next(hf_lm.parameters()).dtype
  bsz = xt.shape[0]
  idx = torch.arange(w0, w1, device=device)
  win_ids = xt[:, w0:w1]

  inner = hf_lm.model
  full_mask = build_eval_sdpa_mask(
      active_len, block_size, device=device, dtype=dtype,
      padding_mask=attention_mask)
  q_mask = _sdpa_mask_for_queries(full_mask, idx)

  hidden = inner.embed_tokens(win_ids)
  dummy = torch.zeros(
      bsz, active_len, hidden.shape[-1], device=device, dtype=hidden.dtype)
  full_pos = torch.arange(active_len, device=device).unsqueeze(0).expand(
      bsz, -1).contiguous()
  cos_full, sin_full = inner.rotary_emb(dummy, full_pos)
  cos_w = cos_full.index_select(-2 if cos_full.dim() == 4 else 1, idx)
  sin_w = sin_full.index_select(-2 if sin_full.dim() == 4 else 1, idx)

  for li, layer in enumerate(inner.layers):
    residual = hidden
    h = layer.input_layernorm(hidden)
    attn = layer.self_attn
    q, k, v = _project_qkv(attn, h)
    q, k = _apply_rope(q, k, cos_w, sin_w)
    k_cache, v_cache = cache.layers[li]
    k_cache = k_cache.clone()
    v_cache = v_cache.clone()
    # Hub: block_cache[:, :, replace_position:replace_position+W] = k
    k_cache[:, :, w0:w1, :] = k
    v_cache[:, :, w0:w1, :] = v
    cache.layers[li] = (k_cache, v_cache)

    k_rep = _repeat_kv(k_cache, attn.num_key_value_groups)
    v_rep = _repeat_kv(v_cache, attn.num_key_value_groups)
    out = F.scaled_dot_product_attention(
        q, k_rep, v_rep, attn_mask=q_mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(h.shape[0], h.shape[1], -1)
    out = attn.o_proj(out)
    hidden = residual + out
    residual = hidden
    h = layer.post_attention_layernorm(hidden)
    hidden = residual + layer.mlp(h)

  hidden = inner.norm(hidden)
  win_logits = hf_lm.lm_head(hidden)
  out = torch.zeros(
      bsz, active_len, win_logits.shape[-1],
      device=device, dtype=win_logits.dtype)
  out[:, w0:w1, :] = win_logits
  return out


__all__ = [
    'DualCache',
    'dual_stream_prefill',
    'dual_stream_replace',
    'single_stream_prefill',
    'single_stream_replace',
    'replace_position_indices',
]

