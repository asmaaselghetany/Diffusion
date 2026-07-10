"""Tier 0: masked / uniform ELBO terms (CPU, torch only)."""

import torch

from discrete_diffusion.losses.block_elbo import (
    masked_block_nll_per_token,
    subs_log_probs,
    uniform_block_nll_per_token,
)


def test_subs_log_probs_normalized_on_masked_positions():
  b, l, v = 2, 6, 20
  mask_id = v - 1
  logits = torch.randn(b, l, v)
  xt = torch.full((b, l), mask_id)
  lp = subs_log_probs(logits, xt, mask_id=mask_id, neg_infinity=-1e6)
  assert torch.allclose(lp.exp().sum(-1), torch.ones(b, l), atol=1e-4)


def test_masked_nll_finite():
  b, l, v = 2, 8, 32
  mask_id = 0
  logits = torch.randn(b, l, v)
  xt = torch.randint(1, v, (b, l))
  xt[0, 0] = mask_id
  x0 = torch.randint(1, v, (b, l))
  lp = subs_log_probs(logits, xt, mask_id, -1e6)
  alpha = torch.full((b, l), 0.5)
  dalpha = torch.full((b, l), -0.5)
  out = masked_block_nll_per_token(lp, x0, alpha, dalpha)
  assert out.shape == (b, l)
  assert torch.isfinite(out).all()


def test_uniform_nll_finite():
  b, l, v = 2, 8, 32
  logits = torch.randn(b, l, v)
  xt = torch.randint(0, v, (b, l))
  x0 = torch.randint(0, v, (b, l))
  lp = torch.log_softmax(logits, dim=-1)
  alpha = torch.full((b, l), 0.5)
  dalpha = torch.full((b, l), -0.5)
  out = uniform_block_nll_per_token(lp, xt, x0, alpha, dalpha, v)
  assert out.shape == (b, l)
  assert torch.isfinite(out).all()


def test_masked_and_uniform_nll_differ():
  torch.manual_seed(0)
  b, l, v = 1, 8, 32
  mask_id = 0
  logits = torch.randn(b, l, v)
  xt = torch.randint(1, v, (b, l))
  xt[0, 2] = mask_id
  x0 = torch.randint(1, v, (b, l))
  alpha = torch.full((b, l), 0.4)
  dalpha = torch.full((b, l), -0.6)
  m = masked_block_nll_per_token(
      subs_log_probs(logits, xt, mask_id, -1e6), x0, alpha, dalpha)
  u = uniform_block_nll_per_token(
      torch.log_softmax(logits, -1), xt, x0, alpha, dalpha, v)
  assert not torch.allclose(m, u)
