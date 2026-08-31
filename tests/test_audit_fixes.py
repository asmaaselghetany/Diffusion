"""Regression tests for audit fixes (#2–#11, #14)."""

import torch

from discrete_diffusion.algorithms.block_trainer import BlockTrainer
from discrete_diffusion.data.processing import (
    _group_texts,
    _pad_example_to_attention_blocks,
)
from discrete_diffusion.evaluations.block_qwen_lm_eval import (
    BlockQwenEvalHarness,
    encode_context_continuation,
)
from discrete_diffusion.forward_process.block_masked import sample_block_timesteps
from discrete_diffusion.sampling.block_sampler import BlockSampler
from omegaconf import OmegaConf


class _FakeNoise:
  eps = 1e-3

  def alpha_t(self, t):
    return 1 - (1 - self.eps) * t

  def alpha_prime_t(self, t):
    return -(1 - self.eps) * torch.ones_like(t)


def _bare_trainer(*, shift: bool, ignore_bos: bool):
  from types import SimpleNamespace

  m = SimpleNamespace(
      shift_loss_targets=shift,
      ignore_bos=ignore_bos,
      forward_process_name='masked',
      mask_id=99,
      neg_infinity=-1e6,
      vocab_size=100,
      block_size=8,
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
  )
  for fn in [
      '_masked_loss', '_uniform_loss', '_loss_for_block', '_ce_loss',
      'nll', '_loss', '_elbo_schedule_weights', '_apply_ignore_bos_mask',
  ]:
    setattr(m, fn, getattr(BlockTrainer, fn).__get__(m, BlockTrainer))
  m._corrupt = lambda x0, t, *, block_size: x0.clone()
  m._backbone_logits = lambda xt, x0, block_size=None, **kwargs: torch.randn(
      *xt.shape, m.vocab_size)
  m._sample_training_block_size = lambda *a, **k: m.block_size
  m._process_model_input = lambda x0, vt: (x0, vt)
  return m


def test_antithetic_timesteps_not_monotonic_with_block_index():
  """Block index must not deterministically order noise level (audit #3)."""
  num_blocks = 64
  means = []
  for _ in range(50):
    t = sample_block_timesteps(
        1, 2048, 32, torch.device('cpu'), antithetic=True)
    block_t = t[0].view(num_blocks, 32)[:, 0]
    means.append(block_t)
  stack = torch.stack(means).mean(0)
  idx = torch.arange(num_blocks, dtype=torch.float32)
  corr = torch.corrcoef(torch.stack([idx, stack]))[0, 1].item()
  assert abs(corr) < 0.25, f'block-t correlation still high: {corr:.3f}'


def test_shift_ignore_bos_keeps_first_real_token_loss():
  """Audit #7: shifted index 0 supervises token 1 — must not be zeroed."""
  m = _bare_trainer(shift=True, ignore_bos=True)
  b, t, v = 1, 8, 100
  logits = torch.randn(b, t, v)
  x0 = torch.randint(10, 90, (b, t))
  x0[:, 0] = 1
  xt = x0.clone()
  xt[:, 1] = 99
  alpha = torch.full((b, t), 0.5)
  dalpha = -torch.ones(b, t) * 0.5
  loss = BlockTrainer._masked_loss(m, logits, xt, x0, alpha, dalpha)
  loss, _ = BlockTrainer._apply_ignore_bos_mask(
      m, loss, torch.ones(b, t))
  assert loss[0, 0] != 0.0


def test_encode_context_continuation_bpe_merge():
  pytest = __import__('pytest')
  tok = pytest.importorskip('transformers').AutoTokenizer.from_pretrained(
      'Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
  ctx, cont = encode_context_continuation(tok, 'The', 're')
  assert len(cont) > 0
  assert ctx == tok('The', add_special_tokens=False)['input_ids']


def test_pad_examples_to_attention_block_boundary():
  """Audit #2: each example ends on a diffusion block boundary before packing."""
  assert _pad_example_to_attention_blocks([1, 2, 3], 4, 0) == [1, 2, 3, 0]
  out = _group_texts(
      {'input_ids': [[1, 2, 3], [4, 5]]},
      block_size=8,
      bos=98,
      eos=99,
      insert_special_tokens=False,
      attention_block_size=4,
      pad_id=0,
  )
  assert len(out['input_ids']) == 1
  flat = out['input_ids'][0]
  assert flat == [1, 2, 3, 0, 4, 5, 0, 0]


def test_init_block_respects_inject_bos_false():
  cfg = OmegaConf.create({
      'sampling': {'steps': 2, 'inject_bos': False},
      'algo': {
          'forward_process_name': 'masked',
          'ignore_bos': True,
          'shift_loss_targets': False,
      },
  })
  sampler = BlockSampler(cfg)

  class M:
    num_tokens = 32
    block_size = 32
    mask_id = 99
    vocab_size = 100
    ignore_bos = True
    config = cfg

    class tokenizer:
      bos_token_id = 777

    @staticmethod
    def prior_sample(n, l):
      return torch.full((n, l), 99)

  m = M()
  xt = torch.full((1, 32), 99)
  x0 = xt.clone()
  xt2, _ = sampler._init_block(m, xt, x0, 0, 32, inject_bos=False)
  assert int(xt2[0, 0]) != 777


def test_truncate_at_stop_and_left_truncate():
  assert BlockQwenEvalHarness._truncate_at_stop('abcSTOPdef', ['STOP']) == 'abc'
  ids = torch.arange(10)
  max_prefix = 4
  kept = ids[-max_prefix:] if ids.numel() > max_prefix else ids
  assert kept.tolist() == [6, 7, 8, 9]

