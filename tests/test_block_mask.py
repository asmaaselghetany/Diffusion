"""CPU unit tests for block attention mask geometry + Qwen hook.

Layer 2 checklist: a hook that stops erroring ≠ a hook that builds the
right mask. These tests assert tensor geometry and that
``block_diff_attention_mask`` actually installs the additive SDPA mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from discrete_diffusion.models.block_mask import (
    build_block_diff_bool_mask,
    build_block_generation_bool_mask,
    build_causal_bool_mask,
    build_sdpa_mask,
)
from discrete_diffusion.models.qwen.attention import block_diff_attention_mask


def test_block_mask_shape():
  n, b = 16, 4
  m = build_block_diff_bool_mask(n, b)
  assert m.shape == (2 * n, 2 * n)


def test_block_mask_differs_from_causal_and_full():
  n, b = 16, 4
  block = build_block_diff_bool_mask(n, b)
  causal = build_causal_bool_mask(2 * n)
  full = torch.ones_like(block, dtype=torch.bool)
  assert not torch.equal(block, causal)
  assert not torch.equal(block, full)
  # Strictly sparser than full bidirectional.
  assert int(block.sum()) < int(full.sum())


def test_block_mask_within_block_xt_bidirectional():
  """Within an xt block, tokens attend both directions (not L2R-only)."""
  n, b = 8, 4
  m = build_block_diff_bool_mask(n, b)
  # xt positions 0..3 are block 0.
  assert m[0, 1].item() and m[1, 0].item()
  assert m[0, 3].item() and m[3, 0].item()
  # Not across xt blocks (block 0 vs block 1).
  assert not m[0, 4].item()
  assert not m[4, 0].item()


def test_block_mask_x0_block_causal_across_finished():
  """x0 half: later blocks see earlier x0 blocks; not the reverse."""
  n, b = 8, 4
  m = build_block_diff_bool_mask(n, b)
  # x0 positions start at n=8. block0: 8..11, block1: 12..15.
  # q in block1 attends kv in block0.
  assert m[12, 8].item()
  # q in block0 does NOT attend kv in block1.
  assert not m[8, 12].item()


def test_block_mask_xt_attends_previous_x0_blocks():
  """xt block k attends x0 blocks < k (offset block-causal), not future x0."""
  n, b = 8, 4
  m = build_block_diff_bool_mask(n, b)
  # xt block1 (pos 4) should attend x0 block0 (pos 8), not x0 block1 (pos 12).
  assert m[4, 8].item()
  assert not m[4, 12].item()
  # xt block0 should not attend any x0 (no finished prior block).
  assert not m[0, 8].item()


def test_block_mask_xt_cannot_attend_same_index_x0():
  """Critical: xt[i] must NOT see x0[i] (would leak the clean answer).

  Offset block-causal only allows *previous* x0 blocks. Same-block /
  same-index xt→x0 is forbidden — so floor reconstruct is NOT
  'failed copy from visible answer'; the answer is not visible.
  """
  n, b = 16, 4
  m = build_block_diff_bool_mask(n, b)
  for i in range(n):
    assert not m[i, n + i].item(), f'xt[{i}] must not attend x0[{i}]'
  # Same block, any x0 in that block: also forbidden.
  for blk in range(n // b):
    xt_sl = slice(blk * b, (blk + 1) * b)
    x0_sl = slice(n + blk * b, n + (blk + 1) * b)
    assert not m[xt_sl, x0_sl].any(), f'xt block {blk} must not attend x0 block {blk}'


def test_block_mask_xt_block0_sees_no_x0():
  n, b = 16, 4
  m = build_block_diff_bool_mask(n, b)
  assert not m[0:b, n:].any()


def test_sdpa_mask_neg_inf_on_disallowed():
  n, b = 8, 4
  sdpa = build_sdpa_mask(n, b, device='cpu', dtype=torch.float32)
  allowed = build_block_diff_bool_mask(n, b)
  assert sdpa.shape == (1, 1, 2 * n, 2 * n)
  # Allowed → 0; disallowed → -inf.
  assert torch.isfinite(sdpa[0, 0][allowed]).all()
  assert (sdpa[0, 0][allowed] == 0).all()
  assert torch.isneginf(sdpa[0, 0][~allowed]).all()


class _Inner(nn.Module):
  """Minimal stand-in for Qwen2Model with HF-style mask updater."""

  def __init__(self):
    super().__init__()
    self.calls = 0

  def _update_causal_mask(self, attention_mask, input_tensor, cache_position,
                          past_key_values, output_attentions: bool = False):
    del attention_mask, cache_position, past_key_values, output_attentions
    self.calls += 1
    # Pretend HF returned a causal additive mask; patch must replace it.
    t = input_tensor.size(1)
    return torch.zeros(1, 1, t, t, device=input_tensor.device,
                       dtype=input_tensor.dtype)


class _Outer(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = _Inner()


def test_padding_mask_blocks_pad_positions():
  from discrete_diffusion.models.block_mask import apply_padding_to_sdpa_mask
  n, b = 8, 4
  base = build_sdpa_mask(n, b, device='cpu', dtype=torch.float32)
  pad = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
  out = apply_padding_to_sdpa_mask(base, pad, n)
  assert out.shape[0] == 1
  # Pad query row must be fully blocked (additive mask uses finfo.min, not -inf).
  assert (out[0, 0, 6] <= -1e30).all()
  # Valid query must still attend somewhere.
  assert (out[0, 0, 0] == 0).any()


def test_attention_hook_respects_padding_mask():
  outer = _Outer()
  n, b = 8, 4
  x = torch.zeros(1, 2 * n, 4)
  pad = torch.ones(1, n, dtype=torch.long)
  pad[0, -2:] = 0
  with block_diff_attention_mask(outer, n, b, x.device, x.dtype, padding_mask=pad):
    out = outer.model._update_causal_mask(
        pad, x, None, None, output_attentions=False)
  assert out.shape[0] == 1
  assert (out[0, 0, n + 6] <= -1e30).all()


def test_attention_hook_installs_block_sdpa_mask():
  """Patched _update_causal_mask must return block SDPA mask, not causal zeros."""
  outer = _Outer()
  n, b = 8, 4
  x = torch.zeros(1, 2 * n, 4)  # fake hidden for dtype/device
  with block_diff_attention_mask(outer, n, b, x.device, x.dtype):
    out = outer.model._update_causal_mask(
        None, x, None, None, output_attentions=False)
  expected = build_sdpa_mask(n, b, device=x.device, dtype=x.dtype)
  assert out.shape == expected.shape
  assert torch.equal(out, expected)
  # Hook restored after context.
  restored = outer.model._update_causal_mask(None, x, None, None)
  assert torch.equal(restored, torch.zeros_like(expected))


def test_attention_hook_fails_loudly_without_updater():
  """4.53-style models without _update_causal_mask must hard-fail (no silent no-op)."""
  bare = nn.Module()
  bare.model = nn.Linear(4, 4)  # no _update_causal_mask
  try:
    with block_diff_attention_mask(bare, 8, 4, torch.device('cpu'), torch.float32):
      pass
  except AttributeError as err:
    assert '_update_causal_mask' in str(err)
    assert 'PYTHONPATH' in str(err) or 'transformers' in str(err)
  else:
    raise AssertionError('expected AttributeError for missing updater')


def test_block_generation_mask_prefix_block_geometry():
  """BlockGen generate: xt sees all prefix; prefix never sees xt."""
  ctx, xt_len, bs = 8, 4, 4
  m = build_block_generation_bool_mask(ctx, xt_len, bs)
  assert m.shape == (ctx + xt_len, ctx + xt_len)
  # First block token (idx 8) attends every prefix token.
  assert bool(m[8, :8].all())
  # Prefix token never attends into the noisy block.
  assert not bool(m[:8, 8:].any())
  # Within noisy block: bidirectional (one block).
  assert m[8, 9].item() and m[9, 8].item()
  # Empty prefix: block-diagonal only.
  m0 = build_block_generation_bool_mask(0, 4, 4)
  assert m0.shape == (4, 4)
  assert bool(m0.all())
