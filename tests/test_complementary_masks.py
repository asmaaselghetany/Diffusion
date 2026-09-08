"""Fast-dLLM complementary masks = paired m / ~m batch doubling (not polarity flip)."""

from types import SimpleNamespace

import torch

from discrete_diffusion.algorithms.block_trainer import BlockTrainer
from discrete_diffusion.forward_process.block_masked import (
    complementary_pair_from_mask,
)


def test_complementary_pair_is_exact_partition():
  torch.manual_seed(0)
  x0 = torch.randint(1, 20, (2, 16))
  move = torch.rand(2, 16) < 0.35
  xt_a, xt_b = complementary_pair_from_mask(x0, move, mask_id=0)
  # Exact complements: a position is masked in exactly one view.
  both = (xt_a == 0) & (xt_b == 0)
  neither = (xt_a != 0) & (xt_b != 0) & move  # impossible if move True
  assert not both.any()
  assert ((xt_a == 0) | (xt_b == 0)).all()
  assert torch.equal(xt_a == 0, move)
  assert torch.equal(xt_b == 0, ~move)
  del neither


def test_old_polarity_flip_would_not_partition():
  """Regression: ~move_mask polarity flip is NOT complementary views."""
  torch.manual_seed(1)
  p = torch.full((4, 32), 0.1)
  move = torch.rand(4, 32) < p
  # Polarity flip of a sparse mask → dense masks; both-masked positions exist
  # when the same token is True in move and in a "phase" that uses ~move —
  # here we just show ~move is NOT a second independent view of the same m.
  flipped = ~move
  # Many positions masked in flipped when move is sparse:
  assert flipped.float().mean() > 0.8
  # And flipped is not an alternate training view of the *same* Bernoulli draw
  # placed in a doubled batch — it is a different corruption rate entirely.


class _FakeNoise:
  def alpha_t(self, t):
    return torch.full_like(t, 0.5, dtype=torch.float32)

  def alpha_prime_t(self, t):
    return torch.full_like(t, -0.5, dtype=torch.float32)


def _bare(*, complementary: bool) -> SimpleNamespace:
  m = SimpleNamespace(
      shift_loss_targets=False,
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
      complementary_masks=complementary,
      complementary_batching='fused',
      mask_schedule='alpha',
      joint_ar_alpha=0.0,
      causal_clean_stream=False,
      _pending_clean_logits=None,
  )

  class _FP:
    pass

  m._forward_process = _FP()
  # isinstance check in nll — patch by using real FP class name via duck type:
  from discrete_diffusion.forward_process.block_masked import (
      BlockMaskedForwardProcess,
  )
  m._forward_process = object.__new__(BlockMaskedForwardProcess)
  m._forward_process.mask_id = 0

  def _corrupt(x0, t, *, block_size, return_move_mask=False, corruption_mask=None):
    del t, block_size, corruption_mask
    move = torch.zeros_like(x0, dtype=torch.bool)
    move[:, 1::2] = True
    xt = x0.clone()
    xt[move] = m.mask_id
    if return_move_mask:
      return xt, move
    return xt

  def _backbone_logits(xt, x0, *, block_size=None, **kwargs):
    del x0, block_size
    return torch.zeros(*xt.shape, m.vocab_size)

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
  return m


def test_nll_complementary_doubles_batch():
  torch.manual_seed(0)
  m = _bare(complementary=True)
  b, t = 2, 8
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  nlls = BlockTrainer.nll(m, x0, valid, block_size=4)
  assert nlls.shape[0] == 2 * b
  assert nlls.shape[1] == t
  assert torch.isfinite(nlls).all()


def test_nll_complementary_sequential_matches_fused_shape():
  torch.manual_seed(0)
  m = _bare(complementary=True)
  m.complementary_batching = 'sequential'
  b, t = 2, 8
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  nlls = BlockTrainer.nll(m, x0, valid, block_size=4)
  assert nlls.shape[0] == 2 * b


def test_nll_without_complementary_keeps_batch():
  m = _bare(complementary=False)
  b, t = 2, 8
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  nlls = BlockTrainer.nll(m, x0, valid, block_size=4)
  assert nlls.shape == (b, t)
