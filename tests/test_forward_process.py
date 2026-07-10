"""Tier 0: per-block forward processes (CPU, no model download)."""

import torch

from discrete_diffusion.forward_process.block_masked import (
    BlockMaskedForwardProcess,
    sample_block_timesteps,
)
from discrete_diffusion.forward_process.block_uniform import BlockUniformForwardProcess


class _FakeTokenizer:
  mask_token_id = 99
  _effective_vocab_size = 50

  def __len__(self):
    return 50


class _FakeSchedule:
  def alpha_t(self, t):
    return 1.0 - t


def test_per_block_timesteps_constant_in_block():
  t = sample_block_timesteps(2, 16, 4, 'cpu')
  assert t.shape == (2, 16)
  for b in range(4):
    sl = slice(b * 4, (b + 1) * 4)
    assert torch.all(t[0, sl] == t[0, b * 4])


def test_antithetic_timesteps_in_unit_interval():
  t = sample_block_timesteps(
      4, 16, 4, 'cpu', sampling_eps=1e-3, antithetic=True)
  assert (t >= 1e-3).all() and (t <= 1.0).all()


def test_block_masked_uses_mask_id_only():
  fp = BlockMaskedForwardProcess(_FakeTokenizer(), _FakeSchedule())
  x0 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
  t = torch.full((1, 8), 0.95)
  xt, p_mask = fp(x0, t, block_size=4)
  assert p_mask.shape == x0.shape
  changed = xt != x0
  if changed.any():
    assert (xt[changed] == 99).all()


def test_block_uniform_uses_vocab_range():
  fp = BlockUniformForwardProcess(_FakeTokenizer(), _FakeSchedule())
  x0 = torch.ones(2, 8, dtype=torch.long) * 5
  t = torch.full((2, 8), 0.99)
  xt = fp(x0, t, block_size=4)
  assert xt.min() >= 0
  assert xt.max() < 50
