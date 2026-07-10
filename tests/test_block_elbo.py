"""CPU unit tests for block ELBO helpers."""

import torch

from discrete_diffusion.losses.block_elbo import (
    masked_block_nll_per_token,
    subs_log_probs,
    uniform_block_nll_per_token,
)


def test_subs_log_probs_fixed_on_unmasked():
  logits = torch.randn(1, 4, 8)
  xt = torch.tensor([[1, 7, 3, 7]])
  lp = subs_log_probs(logits, xt, mask_id=7, neg_infinity=-1e6)
  # unmasked position 0: prob on token 1
  assert lp[0, 0, 1] == 0.0
  assert torch.isfinite(lp[0, 0]).all()


def test_masked_block_nll_shape():
  b, l, v = 2, 8, 16
  log_p = torch.log_softmax(torch.randn(b, l, v), dim=-1)
  x0 = torch.randint(0, v, (b, l))
  alpha = torch.full((b, l), 0.5)
  dalpha = torch.full((b, l), -0.1)
  nll = masked_block_nll_per_token(log_p, x0, alpha, dalpha)
  assert nll.shape == (b, l)


def test_uniform_block_nll_shape():
  b, l, v = 2, 8, 16
  log_p = torch.log_softmax(torch.randn(b, l, v), dim=-1)
  x0 = torch.randint(0, v, (b, l))
  xt = torch.randint(0, v, (b, l))
  alpha = torch.full((b, l), 0.5)
  dalpha = torch.full((b, l), -0.1)
  nll = uniform_block_nll_per_token(log_p, xt, x0, alpha, dalpha, v)
  assert nll.shape == (b, l)
  assert torch.isfinite(nll).all()
