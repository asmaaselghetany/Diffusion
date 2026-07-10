"""CPU unit tests for block attention mask (no GPU)."""

import torch

from discrete_diffusion.models.block_mask import (
    build_block_diff_bool_mask,
    build_causal_bool_mask,
)


def test_block_mask_shape():
  n, b = 16, 4
  m = build_block_diff_bool_mask(n, b)
  assert m.shape == (2 * n, 2 * n)


def test_block_mask_differs_from_causal():
  n, b = 16, 4
  block = build_block_diff_bool_mask(n, b)
  causal = build_causal_bool_mask(2 * n)
  assert not torch.equal(block, causal)


def test_block_mask_within_block_xt_connectivity():
  n, b = 8, 4
  m = build_block_diff_bool_mask(n, b)
  # position 0 (xt half) should attend to position 1 (same block in xt)
  assert m[0, 1].item()
