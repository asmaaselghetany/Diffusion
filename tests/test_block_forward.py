"""CPU unit tests for per-block forward processes."""

from types import SimpleNamespace

import torch

from discrete_diffusion.forward_process.block_masked import (
    BlockMaskedForwardProcess,
    sample_block_timesteps,
)
from discrete_diffusion.forward_process.block_uniform import BlockUniformForwardProcess
from discrete_diffusion.noise_schedules.log_linear import LogLinear


class _Tok:
  mask_token = '[MASK]'
  mask_token_id = 7
  vocab_size = 32

  def __len__(self):
    return self.vocab_size


def _schedule():
  return LogLinear(eps=1e-3)


def test_sample_block_timesteps_constant_within_block():
  t = sample_block_timesteps(2, 16, 4, torch.device('cpu'))
  assert t.shape == (2, 16)
  for row in t:
    assert row[0] == row[1] == row[2] == row[3]
    assert row[4] == row[5] == row[6] == row[7]


def test_masked_forward_uses_mask_id():
  torch.manual_seed(0)
  fp = BlockMaskedForwardProcess(_Tok(), _schedule())
  x0 = torch.randint(0, 32, (1, 8))
  t = torch.full((1, 8), 0.5)
  xt, _ = fp(x0, t, block_size=4)
  assert (xt == 7).any()
  assert (xt == x0).any()


def test_uniform_forward_can_replace_tokens():
  torch.manual_seed(0)
  fp = BlockUniformForwardProcess(_Tok(), _schedule())
  x0 = torch.randint(0, 32, (1, 8))
  t = torch.full((1, 8), 0.5)
  xt = fp(x0, t, block_size=4)
  changed = xt != x0
  assert changed.any()
