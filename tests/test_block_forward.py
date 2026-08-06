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


def test_masked_corruption_rate_matches_theory():
  """P(xt=mask) ≈ 1-α under absorbing (independent of self-token)."""
  torch.manual_seed(0)
  alpha = 0.5
  fp = BlockMaskedForwardProcess(_Tok(), _schedule())
  x0 = torch.randint(0, 32, (256, 64))
  x0 = torch.where(x0 == 7, x0 + 1, x0)  # avoid starting as mask
  noise = _schedule()
  eps = float(noise.eps)
  t_val = (1.0 - alpha) / (1.0 - eps)
  t = torch.full(x0.shape, t_val)
  xt, _ = fp(x0, t, block_size=8)
  rate = (xt == 7).float().mean().item()
  theory = 1.0 - alpha
  assert abs(rate - theory) < 0.03, f'rate={rate} theory={theory}'


def test_uniform_self_corruption_rate_matches_theory():
  """P(xt=x0) ≈ α + (1-α)/V under Unif(V) replacement (not absorbing)."""
  torch.manual_seed(0)
  V = 32
  alpha = 0.5
  fp = BlockUniformForwardProcess(_Tok(), _schedule())
  # Large sample for stable rate.
  x0 = torch.randint(0, V, (512, 64))
  t = torch.full(x0.shape, 0.5)  # LogLinear: need alpha_t(t)=0.5 approximately
  # Force known alpha via schedule: find t such that alpha≈0.5, or
  # call forward internals by using noise schedule.
  noise = _schedule()
  # LogLinear alpha_t(t) = 1 - (1-eps)*t roughly → t=(1-alpha)/(1-eps)
  eps = float(noise.eps)
  t_val = (1.0 - alpha) / (1.0 - eps)
  t = torch.full(x0.shape, t_val)
  xt = fp(x0, t, block_size=8)
  rate = (xt == x0).float().mean().item()
  theory = alpha + (1.0 - alpha) / V
  assert abs(rate - theory) < 0.02, f'rate={rate} theory={theory}'


def test_uniform_not_absorbing_to_mask_only():
  """Uniform must place mass on non-mask replacements, not only mask_id."""
  torch.manual_seed(1)
  fp = BlockUniformForwardProcess(_Tok(), _schedule())
  x0 = torch.randint(0, 32, (64, 32))
  # Avoid starting as mask so replacements are visible.
  x0 = torch.where(x0 == 7, x0 + 1, x0)
  t = torch.ones_like(x0, dtype=torch.float32) * 0.9
  xt = fp(x0, t, block_size=8)
  changed = xt != x0
  assert changed.any()
  # Among changed tokens, many should be non-mask (Unif over V).
  non_mask_changes = changed & (xt != 7)
  assert non_mask_changes.any()
