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
