"""Hub eval_block_diff_mask = single-stream block-causal."""

import torch

from discrete_diffusion.models.block_mask import (
    build_eval_block_bool_mask,
    eval_block_diff_mask,
)


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
