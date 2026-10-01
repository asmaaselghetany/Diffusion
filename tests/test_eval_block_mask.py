"""Hub eval_block_diff_mask = single-stream block-causal."""

import torch

from discrete_diffusion.models.block_mask import (
    build_eval_block_bool_mask,
    build_eval_sdpa_mask,
    eval_block_diff_mask,
)
from discrete_diffusion.models.qwen.attention import eval_block_causal_attention_mask


def test_eval_block_diff_mask_within_and_across_blocks():
  # seq 8, block 4 → blocks [0:4), [4:8)
  q = torch.arange(8)[:, None]
  kv = torch.arange(8)[None, :]
  allow = eval_block_diff_mask(q, kv, block_size=4)
  # Position 5 (block 1) may attend to all of block 0 and block 1.
  assert allow[5, :8].all()
  # Position 2 (block 0) must not attend to block 1.
  assert not allow[2, 4:].any()
  assert allow[2, :4].all()


def test_build_eval_block_bool_mask_cache_offset():
  # Queries for active block of length 4 with 4 tokens already cached.
  allow = build_eval_block_bool_mask(4, block_size=4, cache_seq_len=4)
  assert allow.shape == (4, 8)
  # Query pos 0 of active block = global 4 → block 1; can see all 8 cols.
  assert allow[0].all()


def test_eval_sdpa_mask_respects_padding():
  seq, b = 8, 4
  pad = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
  out = build_eval_sdpa_mask(
      seq, b, device='cpu', dtype=torch.float32, padding_mask=pad)
  assert out.shape == (1, 1, seq, seq)
  # Pad query/key columns fully blocked.
  assert (out[0, 0, 6] <= -1e30).all()
  assert (out[0, 0, :, 6] <= -1e30).all()
  # Valid within-block attention still open (block0: positions 0..3).
  assert (out[0, 0, 0, :4] == 0).all()


def test_eval_attention_hook_respects_padding_mask():
  import torch.nn as nn

  class _Inner(nn.Module):
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position,
                            past_key_values, output_attentions: bool = False):
      del attention_mask, cache_position, past_key_values, output_attentions
      t = input_tensor.size(1)
      return torch.zeros(1, 1, t, t, device=input_tensor.device,
                         dtype=input_tensor.dtype)

  class _Outer(nn.Module):
    def __init__(self):
      super().__init__()
      self.model = _Inner()

  outer = _Outer()
  seq, b = 8, 4
  x = torch.zeros(1, seq, 4)
  pad = torch.ones(1, seq, dtype=torch.long)
  pad[0, -2:] = 0
  with eval_block_causal_attention_mask(
      outer, seq, b, x.device, x.dtype, padding_mask=pad):
    out = outer.model._update_causal_mask(
        pad, x, None, None, output_attentions=False)
  assert out.shape[0] == 1
  assert (out[0, 0, 6] <= -1e30).all()
