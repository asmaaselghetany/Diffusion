"""fast_dllm forward p_mask must match ELBO α_eff = 1 - p."""

from types import SimpleNamespace

import torch

from discrete_diffusion.algorithms.block_trainer import BlockTrainer
from discrete_diffusion.noise_schedules.log_linear import LogLinear


def test_elbo_weights_match_fast_dllm_p_mask():
  m = SimpleNamespace(
      mask_schedule='fast_dllm',
      forward_process_name='masked',
      noise=LogLinear(eps=1e-3),
  )
  t = torch.tensor([[0.0, 0.25, 0.5, 1.0]], dtype=torch.float32)
  alpha, dalpha = BlockTrainer._elbo_schedule_weights(m, t)
  eps = 1e-3
  p_mask = (1.0 - eps) * t + eps
  assert torch.allclose(alpha, 1.0 - p_mask)
  assert torch.allclose(dalpha, torch.full_like(t, -(1.0 - eps)))


def test_elbo_weights_alpha_schedule_is_log_linear():
  noise = LogLinear(eps=1e-3)
  m = SimpleNamespace(
      mask_schedule='alpha',
      forward_process_name='masked',
      noise=noise,
  )
  t = torch.linspace(0, 1, 5).view(1, -1)
  alpha, dalpha = BlockTrainer._elbo_schedule_weights(m, t)
  assert torch.allclose(alpha, noise.alpha_t(t))
  assert torch.allclose(dalpha, noise.alpha_prime_t(t))


def test_fast_dllm_does_not_double_apply_sampling_eps():
  """Hub: t~U(0,1) then p=(1-ε)t+ε. Do not also floor t to [eps,1]."""
  from discrete_diffusion.forward_process.block_masked import (
      BlockMaskedForwardProcess,
      sample_block_timesteps,
  )

  # Floored path (alpha schedule): t >= sampling_eps.
  t_floor = sample_block_timesteps(
      8, 64, 32, torch.device('cpu'), sampling_eps=1e-3, antithetic=False)
  assert float(t_floor.min()) >= 1e-3 - 1e-9

  # Hub-matched path (trainer sets sampling_eps=0 for fast_dllm): raw U.
  # Explicit zeros must be allowed through to FP.
  class _Tok:
    mask_token_id = 0
    vocab_size = 16

  fp = BlockMaskedForwardProcess(_Tok(), LogLinear(eps=1e-3))
  x0 = torch.ones(1, 8, dtype=torch.long)
  t0 = torch.zeros(1, 8)
  _, p = fp(x0, t0, block_size=4, mask_schedule='fast_dllm')
  # Single eps map: p(t=0)=ε, not ~(1-ε)ε+ε ≈ 2ε.
  assert torch.allclose(p, torch.full_like(p, 1e-3))
  assert float(p.min()) < 1.5e-3
