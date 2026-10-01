"""Shift-loss valid_tokens alignment for BlockTrainer (Track 2 prerequisite)."""

from types import SimpleNamespace

import torch

from discrete_diffusion.algorithms.block_trainer import BlockTrainer
from discrete_diffusion.algorithms.base import Loss


class _FakeNoise:
  def alpha_t(self, t):
    return torch.full_like(t, 0.5, dtype=torch.float32)

  def alpha_prime_t(self, t):
    return torch.full_like(t, -0.5, dtype=torch.float32)


def _bare_masked_trainer(*, shift: bool) -> SimpleNamespace:
  m = SimpleNamespace(
      shift_loss_targets=shift,
      ignore_bos=False,
      forward_process_name='masked',
      mask_id=0,
      neg_infinity=-1e6,
      vocab_size=32,
      num_tokens=8,
      block_size=4,
      block_size_mixture=[],
      block_weights=None,
      pure_noise_block_sizes=set(),
      loss_per_block_size={},
      loss_type='elbo',
      device=torch.device('cpu'),
      sampling_eps=1e-3,
      antithetic_sampling=False,
      stratified_gamma=None,
      noise=_FakeNoise(),
      complementary_masks=False,
      mask_schedule='alpha',
      joint_ar_alpha=0.0,
      causal_clean_stream=False,
      _pending_clean_logits=None,
      loss_weighting='elbo',
      _plain_ce_mask_buf=None,
      _plain_ce_token_count=None,
      tokenizer=None,
  )

  def _corrupt(x0, t, *, block_size, corruption_mask=None, **_kwargs):
    del t, block_size, corruption_mask
    xt = x0.clone()
    xt[:, 1::2] = m.mask_id
    return xt

  def _backbone_logits(xt, x0, *, block_size=None, **kwargs):
    del x0, block_size
    return torch.randn(*xt.shape, m.vocab_size)

  m._corrupt = _corrupt
  m._backbone_logits = _backbone_logits
  m._sample_training_block_size = lambda *a, **k: m.block_size
  m._process_model_input = lambda x0, vt: (x0, vt)
  m._masked_loss = lambda *a, **k: BlockTrainer._masked_loss(m, *a, **k)
  m._uniform_loss = lambda *a, **k: BlockTrainer._uniform_loss(m, *a, **k)
  m._loss_for_block = lambda *a, **k: BlockTrainer._loss_for_block(m, *a, **k)
  m._ce_loss = lambda *a, **k: BlockTrainer._ce_loss(m, *a, **k)
  m._apply_ignore_bos_mask = lambda *a, **k: BlockTrainer._apply_ignore_bos_mask(m, *a, **k)
  m.nll = lambda *a, **k: BlockTrainer.nll(m, *a, **k)
  m._elbo_schedule_weights = lambda t: BlockTrainer._elbo_schedule_weights(m, t)
  m._record_plain_ce_mask = lambda mp: BlockTrainer._record_plain_ce_mask(m, mp)
  m._loss = lambda *a, **k: BlockTrainer._loss(m, *a, **k)
  return m


def test_uniform_loss_shift_shortens_to_t_minus_1():
  """Unifusion port: uniform + shift also drops last logit / first label."""
  m = _bare_masked_trainer(shift=True)
  m.forward_process_name = 'uniform'
  b, t, v = 2, 8, 32
  logits = torch.randn(b, t, v)
  xt = torch.randint(1, v, (b, t))
  x0 = torch.randint(1, v, (b, t))
  alpha = torch.full((b, t), 0.5)
  dalpha = torch.full((b, t), -0.5)
  loss = BlockTrainer._uniform_loss(m, logits, xt, x0, alpha, dalpha)
  assert loss.shape == (b, t - 1)


def test_masked_loss_shift_shortens_to_t_minus_1():
  m = _bare_masked_trainer(shift=True)
  b, t, v = 2, 8, 32
  logits = torch.randn(b, t, v)
  xt = torch.randint(1, v, (b, t))
  xt[:, 2] = 0
  x0 = torch.randint(1, v, (b, t))
  alpha = torch.full((b, t), 0.5)
  dalpha = torch.full((b, t), -0.5)
  loss = BlockTrainer._masked_loss(m, logits, xt, x0, alpha, dalpha)
  assert loss.shape == (b, t - 1)


def test_shift_valid_tokens_multiply_aligns():
  """Without the Track-2 trim, loss*(full valid) raises; with trim it works."""
  m = _bare_masked_trainer(shift=True)
  b, t, v = 2, 8, 32
  logits = torch.randn(b, t, v)
  xt = torch.randint(1, v, (b, t))
  x0 = torch.randint(1, v, (b, t))
  alpha = torch.full((b, t), 0.5)
  dalpha = torch.full((b, t), -0.5)
  loss = BlockTrainer._masked_loss(m, logits, xt, x0, alpha, dalpha)
  valid = torch.ones(b, t)
  raised = False
  try:
    _ = loss * valid
  except RuntimeError:
    raised = True
  assert raised, 'expected shape mismatch before trim'
  if (m.shift_loss_targets
      and valid.size(-1) == loss.size(-1) + 1):
    valid = valid[:, 1:]
  out = loss * valid
  assert out.shape == loss.shape


def test_nll_and_loss_with_shift_no_crash():
  torch.manual_seed(0)
  m = _bare_masked_trainer(shift=True)
  b, t = 2, 8
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  nlls = BlockTrainer.nll(m, x0, valid, block_size=4)
  assert nlls.shape == (b, t - 1)
  assert torch.isfinite(nlls).all()
  out = BlockTrainer._loss(m, x0, valid)
  assert isinstance(out, Loss)
  assert out.num_tokens.item() == float(valid[:, 1:].sum())
  assert torch.isfinite(out.loss)


def test_nll_without_shift_keeps_full_length():
  m = _bare_masked_trainer(shift=False)
  b, t = 2, 8
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  nlls = BlockTrainer.nll(m, x0, valid, block_size=4)
  assert nlls.shape == (b, t)


def test_shift_plain_ce_is_minimize_oriented():
  """shift + plain_ce must return positive CE (not -ce).

  C2 (1660576) inverted this and drove trainer/loss -6 → -730 with logit
  collapse; non-shift plain_ce already used masked_plain_ce_per_token.
  """
  m = _bare_masked_trainer(shift=True)
  m.loss_weighting = 'plain_ce'
  b, t, v = 2, 8, 32
  torch.manual_seed(0)
  xt = torch.full((b, t), m.mask_id)
  x0 = torch.randint(1, v, (b, t))
  alpha = torch.full((b, t), 0.5)
  dalpha = torch.full((b, t), -0.5)

  # Confident wrong: mass on token 1, labels are in 2..v-1
  wrong = torch.full((b, t, v), -20.0)
  wrong[..., 1] = 20.0
  loss_wrong = BlockTrainer._masked_loss(m, wrong, xt, x0, alpha, dalpha)

  # Confident correct on the shifted grid (logits[i] → x0[i+1])
  right = torch.full((b, t, v), -20.0)
  for i in range(t - 1):
    right[:, i, :].scatter_(1, x0[:, i + 1:i + 2], 20.0)
  loss_right = BlockTrainer._masked_loss(m, right, xt, x0, alpha, dalpha)

  assert loss_wrong.shape == (b, t - 1)
  assert float(loss_wrong.mean()) > 1.0  # large positive CE
  assert float(loss_right.mean()) < 0.1  # near-zero CE
  assert float(loss_wrong.mean()) > float(loss_right.mean())


def test_plain_ce_num_tokens_is_mask_sites_only():
  """Hub ForCausalLMLoss averages over labels!=-100 (= mask sites), not all T.

  Regression: dividing by valid_tokens.sum() under-scales plain_ce vs Hub
  (and vs ELBO), especially with complementary 2B (~½).
  """
  m = _bare_masked_trainer(shift=True)
  m.loss_weighting = 'plain_ce'
  b, t = 2, 8
  torch.manual_seed(0)
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  out = BlockTrainer._loss(m, x0, valid)
  # Fake corrupt masks xt[:, 1::2]; after shift that is 4 mask sites / row.
  expected_masks = float(b * 4)
  assert out.num_tokens.item() == expected_masks
  # Must NOT equal all valid positions on the T-1 grid (2*7=14).
  assert out.num_tokens.item() != float(valid[:, 1:].sum())
  assert torch.isfinite(out.loss)


def test_plain_ce_mean_matches_mask_only_average():
  """Optimized loss == nll_sum / mask_site_count (Hub CE reduction)."""
  m = _bare_masked_trainer(shift=True)
  m.loss_weighting = 'plain_ce'
  b, t = 2, 8
  torch.manual_seed(1)
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  out = BlockTrainer._loss(m, x0, valid)
  assert out.num_tokens.item() > 0
  assert out.num_tokens.item() < float(valid[:, 1:].sum())
  expected = out.nlls / out.num_tokens.clamp(min=1)
  assert torch.allclose(out.loss, expected, rtol=1e-5, atol=1e-5)


def test_plain_ce_complementary_denom_is_mask_union_not_2b_valid():
  """Complementary cats m/~m on batch; Hub still means over mask labels only.

  Across both views each supervised token is masked in exactly one view, so
  denom ≈ B*(T-1), not 2B*(T-1).
  """
  from discrete_diffusion.forward_process.block_masked import (
      BlockMaskedForwardProcess,
  )

  m = _bare_masked_trainer(shift=True)
  m.loss_weighting = 'plain_ce'
  m.complementary_masks = True
  m.complementary_batching = 'fused'
  # isinstance gate only — corrupt is mocked below.
  m._forward_process = BlockMaskedForwardProcess.__new__(
      BlockMaskedForwardProcess)

  def _corrupt(x0, t, *, block_size, return_move_mask=False, **_kwargs):
    del t, block_size
    # Partition: odd positions in view A, even in view B (after BOS).
    move = torch.zeros_like(x0, dtype=torch.bool)
    move[:, 1::2] = True
    xt = x0.clone()
    xt[move] = m.mask_id
    if return_move_mask:
      return xt, move
    return xt

  m._corrupt = _corrupt

  b, t = 2, 8
  torch.manual_seed(2)
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  out = BlockTrainer._loss(m, x0, valid)
  # Shift grid T-1=7; move masks odds 1,3,5,7 → 4; complement masks 2,4,6
  # (original 0 was ~move but dropped by shift). Total = 4+3 = 7 per row
  # × B = 14 — equals one full shifted sequence, NOT 2B*(T-1)=28.
  assert out.num_tokens.item() == float(b * 7)
  doubled_valid = float(2 * b * (t - 1))
  assert out.num_tokens.item() != doubled_valid
  assert torch.isfinite(out.loss)
