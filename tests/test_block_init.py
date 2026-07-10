"""Tier 0: AR→block init metric math."""

import torch

from discrete_diffusion.training.init import logit_agreement_rho, top1_agreement


def test_logit_agreement_identical():
  logits = torch.randn(2, 4, 16)
  assert logit_agreement_rho(logits, logits) == 1.0
  assert top1_agreement(logits, logits) == 1.0


def test_logit_agreement_differs():
  a = torch.zeros(1, 2, 4)
  b = torch.zeros(1, 2, 4)
  a[..., 0] = 10.0
  b[..., 1] = 10.0
  assert top1_agreement(a, b) == 0.0
  assert logit_agreement_rho(a, b) < 1.0
