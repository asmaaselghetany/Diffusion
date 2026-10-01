"""DualCache prefill / replace_position splice correctness."""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from discrete_diffusion.models.qwen.dual_cache import (
    DualCache,
    dual_stream_prefill,
    dual_stream_replace,
    replace_position_indices,
)
from discrete_diffusion.models.qwen.modeling import shared_block_position_ids


def _tiny_qwen():
  cfg = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B')
  cfg.num_hidden_layers = 2
  cfg.hidden_size = 64
  cfg.intermediate_size = 128
  cfg.num_attention_heads = 4
  cfg.num_key_value_heads = 2
  cfg.vocab_size = 128
  return AutoModelForCausalLM.from_config(cfg)


def test_replace_position_indices():
  idx = replace_position_indices(8, 2, 5, torch.device('cpu'))
  assert torch.equal(idx[:3], torch.tensor([2, 3, 4]))
  assert torch.equal(idx[3:], torch.tensor([10, 11, 12]))


def test_dual_cache_noop_replace_matches_prefill_window():
  """Replace with identical tokens → window logits match prefill."""
  torch.manual_seed(0)
  m = _tiny_qwen()
  m.eval()
  b, a, bs = 1, 8, 4
  xt = torch.randint(1, 50, (b, a))
  x0 = xt.clone()
  x_in = torch.cat([xt, x0], dim=-1)
  pos = shared_block_position_ids(a, 'cpu', batch_size=b)
  logits0, cache = dual_stream_prefill(
      m, x_in, active_len=a, block_size=bs, position_ids=pos)
  w0, w1 = 4, 8
  logits1 = dual_stream_replace(
      m, xt, x0, active_len=a, window=(w0, w1), block_size=bs, cache=cache)
  diff = (logits0[:, w0:w1] - logits1[:, w0:w1]).abs().max().item()
  assert diff < 1e-4, f'noop replace max|Δ|={diff}'


def test_dual_cache_mutated_window_finite():
  torch.manual_seed(1)
  m = _tiny_qwen()
  m.eval()
  b, a, bs = 1, 8, 4
  xt = torch.randint(1, 50, (b, a))
  x0 = xt.clone()
  x_in = torch.cat([xt, x0], dim=-1)
  pos = shared_block_position_ids(a, 'cpu', batch_size=b)
  _, cache = dual_stream_prefill(
      m, x_in, active_len=a, block_size=bs, position_ids=pos)
  xt2 = xt.clone()
  xt2[:, 4:8] = torch.randint(1, 50, (b, 4))
  x0_2 = xt2.clone()
  logits = dual_stream_replace(
      m, xt2, x0_2, active_len=a, window=(4, 8), block_size=bs, cache=cache)
  assert torch.isfinite(logits).all()
  assert logits[:, 4:8].abs().sum() > 0
  assert isinstance(cache, DualCache)


def test_dual_cache_window_replace_matches_full_forward():
  """Single-window DualCache splice must match a full dual-stream forward.

  (Previously asserted ``delta >= 0``, which can never fail.)
  """
  from discrete_diffusion.models.qwen.attention import block_diff_attention_mask

  torch.manual_seed(2)
  m = _tiny_qwen()
  m.eval()
  b, a, bs = 1, 8, 4
  xt = torch.randint(1, 50, (b, a))
  x0 = xt.clone()
  pos = shared_block_position_ids(a, 'cpu', batch_size=b)
  x_in = torch.cat([xt, x0], dim=-1)
  _, cache = dual_stream_prefill(
      m, x_in, active_len=a, block_size=bs, position_ids=pos)

  xt2 = xt.clone()
  xt2[:, 4:8] = torch.randint(1, 50, (b, 4))
  x0_2 = xt2.clone()
  dc = dual_stream_replace(
      m, xt2, x0_2, active_len=a, window=(4, 8), block_size=bs, cache=cache)

  x_full = torch.cat([xt2, x0_2], dim=-1)
  dtype = next(m.parameters()).dtype
  with block_diff_attention_mask(m, a, bs, x_full.device, dtype):
    full = m(input_ids=x_full, position_ids=pos, use_cache=False).logits[:, :a, :]

  assert torch.isfinite(dc[:, 4:8]).all()
  assert torch.allclose(
      dc[:, 4:8], full[:, 4:8].detach(), rtol=1e-4, atol=1e-4), (
      'DualCache window replace diverged from full forward')
