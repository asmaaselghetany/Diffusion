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


def test_size1_eval_uses_ce_not_elbo():
  """BlockGen-aligned: eval at block_size=1 must use CE, not continuous ELBO."""
  from discrete_diffusion.algorithms.block_trainer import BlockTrainer

  class Stub:
    forward_process_name = 'uniform'
    loss_type = 'elbo'
    loss_per_block_size = {}
    vocab_size = 16
    mask_id = 7
    neg_infinity = -1e9
    shift_loss_targets = False

    def _ce_loss(self, logits, xt, x0, *, noisy_only):
      return torch.full(logits.shape[:2], 1.23)

    def _uniform_loss(self, logits, xt, x0, alpha_t, dalpha_t):
      return torch.full(logits.shape[:2], 9.87)

    def _hybrid_loss(self, *a, **k):
      raise AssertionError('hybrid should not run')

    def _masked_loss(self, *a, **k):
      raise AssertionError('masked should not run')

  s = Stub()
  logits = torch.randn(2, 8, 16)
  xt = torch.randint(0, 16, (2, 8))
  x0 = torch.randint(0, 16, (2, 8))
  a = torch.full((2, 8), 0.5)
  da = torch.full((2, 8), -0.1)
  assert abs(float(
      BlockTrainer._loss_for_block(
          s, logits, xt, x0, a, da, block_size=1, train_mode=False).mean()
  ) - 1.23) < 1e-5
  assert abs(float(
      BlockTrainer._loss_for_block(
          s, logits, xt, x0, a, da, block_size=1, train_mode=True).mean()
  ) - 9.87) < 1e-5
  assert abs(float(
      BlockTrainer._loss_for_block(
          s, logits, xt, x0, a, da, block_size=32, train_mode=False).mean()
  ) - 9.87) < 1e-5
