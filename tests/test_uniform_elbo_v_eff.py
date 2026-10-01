"""Uniform ELBO V_eff must match a compact (MASK-dropped) Duo sum."""

from __future__ import annotations

import torch

from discrete_diffusion.losses.block_elbo import uniform_block_nll_per_token


def test_uniform_elbo_mask_id_matches_compact_v_eff():
  torch.manual_seed(0)
  b, length, v = 3, 5, 20
  mask_id = 7
  v_eff = v - 1
  logits = torch.randn(b, length, v)
  logits[..., mask_id] = -1e9
  log_p = torch.log_softmax(logits, dim=-1)
  keep = [i for i in range(v) if i != mask_id]
  log_p_c = log_p[..., keep]
  # Compact ids → full ids (never MASK).
  id_map = torch.tensor(keep)
  xt_c = torch.randint(0, v_eff, (b, length))
  x0_c = torch.randint(0, v_eff, (b, length))
  xt = id_map[xt_c]
  x0 = id_map[x0_c]
  alpha = torch.rand(b, length).clamp(0.05, 0.95)
  dalpha = torch.rand(b, length) * 0.1

  broken = uniform_block_nll_per_token(
      log_p, xt, x0, alpha, dalpha, v_eff)  # no mask_id
  fixed = uniform_block_nll_per_token(
      log_p, xt, x0, alpha, dalpha, v_eff, mask_id=mask_id)
  compact = uniform_block_nll_per_token(
      log_p_c, xt_c, x0_c, alpha, dalpha, v_eff)

  assert torch.allclose(fixed, compact, rtol=1e-4, atol=1e-5), (
      float((fixed - compact).abs().max()),
      float((broken - compact).abs().max()))
  # Broken path should differ from compact (the bug we fixed).
  assert not torch.allclose(broken, compact, rtol=1e-4, atol=1e-5)
