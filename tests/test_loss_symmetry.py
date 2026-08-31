"""Layer 3 — masked ↔ uniform loss special-case symmetry audit."""

import inspect

import pytest

from discrete_diffusion.algorithms import block_trainer as bt
from discrete_diffusion.algorithms.block_trainer import (
    LOSS_SPECIAL_CASE_POLICY,
    BlockTrainer,
)


REQUIRED_POLICY_KEYS = {
    'shift_loss_targets',
    'complementary_masks',
    'mask_schedule',
    'subs_log_probs',
    'ignore_bos',
    'valid_tokens_shift_trim',
    'pure_noise_block_sizes',
    'loss_type_special_cases',
    'block_weights',
    'joint_ar_alpha',
    'causal_clean_stream',
    'hybrid_p_uniform',
}


def test_loss_special_case_policy_complete():
  assert REQUIRED_POLICY_KEYS <= set(LOSS_SPECIAL_CASE_POLICY)
  allowed = {
      'masked_only', 'shared', 'masked_only_when_shift', 'applies',
      'hybrid_only',
  }
  for k, v in LOSS_SPECIAL_CASE_POLICY.items():
    assert v in allowed, f'{k}={v} not in {allowed}'


def test_uniform_loss_does_not_implement_masked_only_hooks():
  """Source audit: _uniform_loss must not silently grow shift/SUBS paths."""
  src = inspect.getsource(BlockTrainer._uniform_loss)
  # Documented non-applications may mention these names in the docstring;
  # executable body must not call shift / subs helpers.
  body = src.split('"""', 2)[-1] if '"""' in src else src
  assert 'shift_loss_targets' not in body
  assert 'subs_log_probs' not in body
  assert 'mask_id' not in body  # uniform scores all sites via DUO


def test_masked_loss_owns_shift_and_subs():
  src = inspect.getsource(BlockTrainer._masked_loss)
  assert 'shift_loss_targets' in src
  assert 'subs_log_probs' in src


def test_ignore_bos_shared_in_nll():
  src = inspect.getsource(BlockTrainer.nll)
  assert 'ignore_bos' in src
  assert '_apply_ignore_bos_mask' in src
  assert LOSS_SPECIAL_CASE_POLICY['ignore_bos'] == 'shared'


def test_policy_exported_for_docs():
  assert hasattr(bt, 'LOSS_SPECIAL_CASE_POLICY')
  assert bt.LOSS_SPECIAL_CASE_POLICY is LOSS_SPECIAL_CASE_POLICY


def _fake_trainer_base_init(self, config, tokenizer, vocab_size=None):
  """Skip HF backbone; only exercise BlockTrainer validate/policy."""
  import lightning as L
  import torch
  from discrete_diffusion.noise_schedules.log_linear import LogLinear

  L.LightningModule.__init__(self)
  self.config = config
  self.tokenizer = tokenizer
  self.vocab_size = vocab_size or len(tokenizer)
  self.ignore_bos = True
  self.loss_type = 'elbo'
  self.sampler = 'ddpm'
  self.antithetic_sampling = False
  self.parameterization = 'subs'
  self._sampler_cfg = None
  self._sampler = None
  self.backbone = object()
  self.model = self.backbone
  self.T = 0
  self.num_tokens = 16
  self.noise = LogLinear(eps=1e-3)
  self.time_conditioning = False
  self.neg_infinity = -1e6
  self.sampling_eps = 1e-3


class _TinyTok:
  mask_token = '[MASK]'
  mask_token_id = 7
  pad_token_id = 0
  eos_token_id = 1
  vocab_size = 32

  def __len__(self):
    return 32


def _uniform_cfg(**algo_overrides):
  import omegaconf
  cfg = omegaconf.OmegaConf.create({
      'algo': {
          'forward_process_name': 'uniform',
          'parameterization': 'subs',
          'time_conditioning': False,
          'T': 0,
          'loss_type': 'elbo',
          'ignore_bos': True,
          'shift_loss_targets': False,
          'complementary_masks': False,
          'mask_schedule': 'alpha',
          'block_size_mixture': [],
          'block_weights': None,
          'block_size_per_gpu': None,
          'stratified_gamma': None,
          'pure_noise_block_sizes': [],
          'loss_type_special_cases': [],
      },
      'model': {'length': 16},
      'block_size': 4,
      'sampling': {'predictor': 'ddpm', 'use_arpc': False},
      'training': {'antithetic_sampling': False},
      'seed': 0,
  })
  for k, v in algo_overrides.items():
    cfg.algo[k] = v
  return cfg


def test_uniform_shift_refuses_at_trainer_init():
  """Hard ValueError — not silent ignore (shift-loss surprise class)."""
  from unittest.mock import patch
  import discrete_diffusion.algorithms.base as base_mod
  with patch.object(base_mod.TrainerBase, '__init__', _fake_trainer_base_init):
    with pytest.raises(ValueError, match='shift_loss_targets.*masked-only'):
      BlockTrainer(_uniform_cfg(shift_loss_targets=True), _TinyTok())


def test_uniform_complementary_refuses_at_trainer_init():
  from unittest.mock import patch
  import discrete_diffusion.algorithms.base as base_mod
  with patch.object(base_mod.TrainerBase, '__init__', _fake_trainer_base_init):
    with pytest.raises(ValueError, match='complementary_masks.*masked-only'):
      BlockTrainer(_uniform_cfg(complementary_masks=True), _TinyTok())


def test_t_bucketed_nll_shift_trim_aligns():
  """Regression 138103: shift loss T-1 must trim valid_tokens in t_bucketed."""
  from unittest.mock import MagicMock, patch
  import torch
  import discrete_diffusion.algorithms.base as base_mod

  m = _fake_trainer_base_init  # reuse shape only via bare ns
  from types import SimpleNamespace
  trainer = SimpleNamespace(
      shift_loss_targets=True,
      ignore_bos=False,
      forward_process_name='masked',
      complementary_masks=False,
      mask_schedule='alpha',
      mask_id=0,
      neg_infinity=-1e6,
      vocab_size=32,
      block_size=4,
      loss_type='elbo',
      loss_per_block_size={},
      device=torch.device('cpu'),
      noise=type('N', (), {
          'eps': 1e-3,
          'alpha_t': staticmethod(lambda t: torch.full_like(t, 0.5)),
          'alpha_prime_t': staticmethod(lambda t: torch.full_like(t, -0.5)),
      })(),
  )
  trainer._process_model_input = lambda x0, vt: (x0, vt)
  trainer._corrupt = lambda x0, t, block_size: x0.clone()
  trainer._backbone_logits = lambda xt, x0, block_size=None, **kwargs: torch.randn(
      *xt.shape, trainer.vocab_size)
  trainer._masked_loss = lambda *a, **k: BlockTrainer._masked_loss(trainer, *a, **k)
  trainer._uniform_loss = lambda *a, **k: BlockTrainer._uniform_loss(trainer, *a, **k)
  trainer._loss_for_block = lambda *a, **k: BlockTrainer._loss_for_block(trainer, *a, **k)
  trainer._ce_loss = lambda *a, **k: BlockTrainer._ce_loss(trainer, *a, **k)
  trainer._apply_ignore_bos_mask = lambda *a, **k: BlockTrainer._apply_ignore_bos_mask(trainer, *a, **k)
  trainer._elbo_schedule_weights = lambda t: BlockTrainer._elbo_schedule_weights(
      trainer, t)
  trainer.log = MagicMock()
  b, t = 2, 8
  x0 = torch.randint(1, 32, (b, t))
  vt = torch.ones(b, t)
  # Must not raise 511-vs-512 style broadcast error.
  BlockTrainer._log_t_bucketed_nll(trainer, x0, vt)
  assert trainer.log.called
