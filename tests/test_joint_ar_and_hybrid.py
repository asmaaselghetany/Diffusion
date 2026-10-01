"""C5 joint AR loss combine + B4 hybrid forward process."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from discrete_diffusion.algorithms.block_trainer import BlockTrainer
from discrete_diffusion.forward_process.block_hybrid import BlockHybridForwardProcess
from discrete_diffusion.forward_process.block_masked import BlockMaskedForwardProcess
from discrete_diffusion.noise_schedules.log_linear import LogLinear


class _Tok:
  mask_token = '[MASK]'
  mask_token_id = 7
  pad_token_id = 0
  eos_token_id = 1
  vocab_size = 32

  def __len__(self):
    return 32


def _schedule():
  return LogLinear(eps=1e-3)


class _FakeNoise:
  def alpha_t(self, t):
    return torch.full_like(t, 0.5, dtype=torch.float32)

  def alpha_prime_t(self, t):
    return torch.full_like(t, -0.5, dtype=torch.float32)


def _bare_joint(*, alpha: float, causal: bool = False) -> SimpleNamespace:
  m = SimpleNamespace(
      shift_loss_targets=False,
      ignore_bos=True,
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
      joint_ar_alpha=alpha,
      causal_clean_stream=causal,
      loss_weighting='elbo',
      hub_struct_attn_only=False,
      _pending_clean_logits=None,
      _plain_ce_mask_buf=None,
      _plain_ce_token_count=None,
      _last_ar_nll=None,
      _last_diff_nll=None,
  )
  m._forward_process = object.__new__(BlockMaskedForwardProcess)
  m._forward_process.mask_id = 0

  def _corrupt(x0, t, *, block_size, return_move_mask=False, corruption_mask=None):
    del t, block_size, return_move_mask, corruption_mask
    return x0.clone()

  def _backbone_logits(xt, x0, *, block_size=None, return_clean=False, **kwargs):
    del block_size
    logits = torch.zeros(*xt.shape, m.vocab_size)
    # Prefer correct next token lightly so CE is finite / non-trivial.
    logits.scatter_(-1, x0.unsqueeze(-1), 2.0)
    if return_clean:
      clean = logits.clone()
      return logits, clean
    return logits

  class _Backbone:
    def causal_train_logits(self, x0):
      logits = torch.zeros(*x0.shape, m.vocab_size)
      logits.scatter_(-1, x0.unsqueeze(-1), 2.0)
      return logits

  m.backbone = _Backbone()
  m._corrupt = _corrupt
  m._backbone_logits = _backbone_logits
  m._sample_training_block_size = lambda *a, **k: m.block_size
  m._process_model_input = lambda x0, vt: (x0, vt)
  m.nll = lambda *a, **k: BlockTrainer.nll(m, *a, **k)
  m._masked_loss = lambda *a, **k: BlockTrainer._masked_loss(m, *a, **k)
  m._uniform_loss = lambda *a, **k: BlockTrainer._uniform_loss(m, *a, **k)
  m._hybrid_loss = lambda *a, **k: BlockTrainer._hybrid_loss(m, *a, **k)
  m._loss_for_block = lambda *a, **k: BlockTrainer._loss_for_block(m, *a, **k)
  m._ce_loss = lambda *a, **k: BlockTrainer._ce_loss(m, *a, **k)
  m._ar_ce_from_logits = lambda *a, **k: BlockTrainer._ar_ce_from_logits(m, *a, **k)
  m._ar_ce_mean = lambda *a, **k: BlockTrainer._ar_ce_mean(m, *a, **k)
  m._apply_ignore_bos_mask = lambda *a, **k: BlockTrainer._apply_ignore_bos_mask(m, *a, **k)
  m._elbo_schedule_weights = lambda t: BlockTrainer._elbo_schedule_weights(m, t)
  return m


def test_joint_ar_alpha_zero_matches_diff_only():
  torch.manual_seed(0)
  m = _bare_joint(alpha=0.0)
  b, t = 2, 8
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  loss0 = BlockTrainer._loss(m, x0, valid)
  m.joint_ar_alpha = 0.3
  # Force clean logits path
  loss_a = BlockTrainer._loss(m, x0, valid)
  assert torch.isfinite(loss0.loss)
  assert torch.isfinite(loss_a.loss)
  # Combined should differ from pure diff when alpha>0
  assert not torch.isclose(loss0.loss, loss_a.loss)


def test_joint_ar_combine_formula():
  torch.manual_seed(1)
  m = _bare_joint(alpha=0.3)
  b, t = 2, 8
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  out = BlockTrainer._loss(m, x0, valid)
  assert m._last_ar_nll is not None
  assert m._last_diff_nll is not None
  expected = m._last_ar_nll + 0.3 * m._last_diff_nll
  assert torch.isclose(out.loss, expected, atol=1e-5)


def test_causal_clean_uses_causal_path():
  torch.manual_seed(2)
  m = _bare_joint(alpha=0.3, causal=True)
  b, t = 2, 8
  x0 = torch.randint(1, m.vocab_size, (b, t))
  valid = torch.ones(b, t)
  out = BlockTrainer._loss(m, x0, valid)
  assert torch.isfinite(out.loss)
  assert m._pending_clean_logits is None  # causal path; no dual clean stash


def test_hybrid_p0_matches_mask_rate():
  torch.manual_seed(0)
  alpha = 0.5
  fp = BlockHybridForwardProcess(_Tok(), _schedule(), p_uniform=0.0)
  x0 = torch.randint(0, 32, (256, 64))
  x0 = torch.where(x0 == 7, x0 + 1, x0)
  noise = _schedule()
  eps = float(noise.eps)
  t_val = (1.0 - alpha) / (1.0 - eps)
  t = torch.full(x0.shape, t_val)
  xt = fp(x0, t, block_size=8)
  rate = (xt == 7).float().mean().item()
  assert abs(rate - (1.0 - alpha)) < 0.03


def test_hybrid_p1_rarely_mask():
  torch.manual_seed(0)
  alpha = 0.5
  fp = BlockHybridForwardProcess(_Tok(), _schedule(), p_uniform=1.0)
  x0 = torch.randint(0, 32, (256, 64))
  x0 = torch.where(x0 == 7, x0 + 1, x0)
  noise = _schedule()
  eps = float(noise.eps)
  t_val = (1.0 - alpha) / (1.0 - eps)
  t = torch.full(x0.shape, t_val)
  xt = fp(x0, t, block_size=8)
  mask_rate = (xt == 7).float().mean().item()
  change_rate = (xt != x0).float().mean().item()
  assert mask_rate < 0.02
  assert abs(change_rate - (1.0 - alpha) * (31 / 31)) < 0.05  # exclude mask


def test_hybrid_p10_mix():
  torch.manual_seed(0)
  alpha = 0.5
  p_u = 0.1
  fp = BlockHybridForwardProcess(_Tok(), _schedule(), p_uniform=p_u)
  x0 = torch.randint(0, 32, (512, 64))
  x0 = torch.where(x0 == 7, x0 + 1, x0)
  noise = _schedule()
  eps = float(noise.eps)
  t_val = (1.0 - alpha) / (1.0 - eps)
  t = torch.full(x0.shape, t_val)
  xt = fp(x0, t, block_size=8)
  moved = xt != x0
  among_moved_mask = ((xt == 7) & moved).float().sum() / moved.float().sum().clamp(min=1)
  # Among replacements, ~90% MASK
  assert abs(among_moved_mask.item() - (1.0 - p_u)) < 0.05


def _fake_trainer_base_init(self, config, tokenizer, vocab_size=None):
  import lightning as L
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


def test_hybrid_shift_refused():
  import omegaconf
  import discrete_diffusion.algorithms.base as base_mod

  cfg = omegaconf.OmegaConf.create({
      'algo': {
          'forward_process_name': 'hybrid',
          'parameterization': 'subs',
          'time_conditioning': False,
          'T': 0,
          'loss_type': 'elbo',
          'ignore_bos': True,
          'shift_loss_targets': True,
          'shift_on_hybrid': False,
          'complementary_masks': False,
          'mask_schedule': 'alpha',
          'block_size_mixture': [],
          'block_weights': None,
          'block_size_per_gpu': None,
          'stratified_gamma': None,
          'pure_noise_block_sizes': [],
          'loss_type_special_cases': [],
          'hybrid_p_uniform': 0.1,
          'joint_ar_alpha': 0.0,
          'causal_clean_stream': False,
          'hybrid_decode': 'masked',
          'intra_block_attn_anneal_steps': 0,
          'kernel_anneal_steps': 0,
      },
      'model': {'length': 16},
      'block_size': 4,
      'sampling': {'predictor': 'ddpm', 'use_arpc': False},
      'training': {'antithetic_sampling': False},
      'seed': 0,
  })
  with patch.object(base_mod.TrainerBase, '__init__', _fake_trainer_base_init):
    with pytest.raises(ValueError, match='shift_on_hybrid'):
      BlockTrainer(cfg, _Tok())


def test_hybrid_shift_on_hybrid_forward_finite():
  """B4_shift_explorative: shared T−1 grid; finite hybrid loss (P0 regression)."""
  from types import SimpleNamespace
  m = SimpleNamespace(
      mask_id=7,
      neg_infinity=-1e6,
      vocab_size=32,
      forward_process_name='hybrid',
      shift_loss_targets=True,
      loss_weighting='elbo',
      tokenizer=None,
  )
  m._record_plain_ce_mask = lambda *_a, **_k: None
  m._masked_loss = lambda *a, **k: BlockTrainer._masked_loss(m, *a, **k)
  b, t, v = 2, 8, 32
  torch.manual_seed(1)
  logits = torch.randn(b, t, v)
  x0 = torch.randint(0, v, (b, t))
  x0 = torch.where(x0 == 7, x0 + 1, x0)
  xt = x0.clone()
  xt[:, 2] = 7
  xt[:, 4] = (x0[:, 4] + 3) % v
  xt[:, 4] = torch.where(xt[:, 4] == 7, xt[:, 4] + 1, xt[:, 4])
  alpha = torch.full((b, t), 0.5)
  dalpha = torch.full((b, t), -0.5)
  loss = BlockTrainer._hybrid_loss(m, logits, xt, x0, alpha, dalpha)
  assert loss.shape == (b, t - 1)
  assert torch.isfinite(loss).all()
  # Shifted grid: planted mask/unif at cols 2/4 still land on the T−1 loss.
  assert (loss[:, 1].abs() > 0).all()  # was xt[:, 2] mask
  assert (loss[:, 3].abs() > 0).all()  # was xt[:, 4] unif


def test_uniform_allows_shift_unifusion_port():
  """Unifusion-style: shift_loss_targets OK on uniform (hybrid still refused)."""
  import omegaconf
  import discrete_diffusion.algorithms.base as base_mod

  cfg = omegaconf.OmegaConf.create({
      'algo': {
          'forward_process_name': 'uniform',
          'parameterization': 'subs',
          'time_conditioning': False,
          'T': 0,
          'loss_type': 'elbo',
          'ignore_bos': True,
          'shift_loss_targets': True,
          'complementary_masks': False,
          'mask_schedule': 'alpha',
          'block_size_mixture': [],
          'block_weights': None,
          'block_size_per_gpu': None,
          'stratified_gamma': None,
          'pure_noise_block_sizes': [],
          'loss_type_special_cases': [],
          'hybrid_p_uniform': 0.1,
          'joint_ar_alpha': 0.0,
          'causal_clean_stream': False,
          'loss_weighting': 'elbo',
          'hub_struct_attn_only': False,
          'hybrid_decode': 'masked',
      },
      'model': {'length': 16},
      'block_size': 4,
      'sampling': {'predictor': 'ddpm', 'use_arpc': False},
      'training': {'antithetic_sampling': False},
      'seed': 0,
  })
  with patch.object(base_mod.TrainerBase, '__init__', _fake_trainer_base_init):
    m = BlockTrainer(cfg, _Tok())
  assert m.shift_loss_targets is True
  assert m.forward_process_name == 'uniform'


def test_hybrid_loss_uses_v_minus_1_and_routes_sites():
  """Unif sites use V-1 DUO; mask sites use masked; clean → 0."""
  from types import SimpleNamespace
  m = SimpleNamespace(
      mask_id=7,
      neg_infinity=-1e6,
      vocab_size=32,
      forward_process_name='hybrid',
      shift_loss_targets=False,
      tokenizer=None,
  )
  m._masked_loss = lambda *a, **k: BlockTrainer._masked_loss(m, *a, **k)
  m._uniform_loss = lambda *a, **k: BlockTrainer._uniform_loss(m, *a, **k)
  b, t, v = 2, 8, 32
  torch.manual_seed(0)
  logits = torch.randn(b, t, v)
  x0 = torch.randint(0, v, (b, t))
  x0 = torch.where(x0 == 7, x0 + 1, x0)
  xt = x0.clone()
  # Plant one mask site and one unif site per row.
  xt[:, 2] = 7
  xt[:, 4] = (x0[:, 4] + 3) % v
  xt[:, 4] = torch.where(xt[:, 4] == 7, xt[:, 4] + 1, xt[:, 4])
  alpha = torch.full((b, t), 0.5)
  dalpha = torch.full((b, t), -0.5)
  loss = BlockTrainer._hybrid_loss(m, logits, xt, x0, alpha, dalpha)
  assert loss.shape == (b, t)
  assert torch.isfinite(loss).all()
  # Clean sites (except planted) should be ~0
  clean = (xt == x0)
  assert (loss[clean].abs() < 1e-5).all()
  assert (loss[:, 2].abs() > 0).all()
  assert (loss[:, 4].abs() > 0).all()


def test_single_stream_train_refuses_joint_ar_without_causal_clean():
  import omegaconf
  import discrete_diffusion.algorithms.base as base_mod

  cfg = omegaconf.OmegaConf.create({
      'algo': {
          'forward_process_name': 'uniform',
          'parameterization': 'mean',
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
          'hybrid_p_uniform': 0.1,
          'joint_ar_alpha': 0.3,
          'causal_clean_stream': False,
          'single_stream_train': True,
          'loss_weighting': 'elbo',
          'hub_struct_attn_only': False,
          'hybrid_decode': 'masked',
      },
      'model': {'length': 16},
      'block_size': 4,
      'sampling': {'predictor': 'ddpm', 'use_arpc': False},
      'training': {'antithetic_sampling': False},
      'seed': 0,
  })
  with patch.object(base_mod.TrainerBase, '__init__', _fake_trainer_base_init):
    with pytest.raises(ValueError, match='single_stream_train'):
      BlockTrainer(cfg, _Tok())


def test_backbone_logits_single_stream_calls_block_train_logits():
  """Packing switch: no concat; uses backbone.block_train_logits."""
  calls = {}

  class _BB:
    def block_train_logits(self, xt, *, active_len=None, block_size=None,
                           attention_mask=None):
      calls['xt'] = xt
      calls['active_len'] = active_len
      calls['block_size'] = block_size
      return torch.zeros(xt.shape[0], active_len, 8)

    def __call__(self, *a, **k):
      raise AssertionError('dual-stream backbone must not be called')

  m = SimpleNamespace(
      single_stream_train=True,
      backbone=_BB(),
  )
  xt = torch.randint(0, 8, (2, 6))
  x0 = torch.randint(0, 8, (2, 6))
  out = BlockTrainer._backbone_logits(m, xt, x0, block_size=4)
  assert out.shape == (2, 6, 8)
  assert calls['active_len'] == 6
  assert calls['block_size'] == 4
  assert torch.equal(calls['xt'], xt)


def test_backbone_logits_dual_stream_concats():
  class _BB:
    def __call__(self, x_in, sigma=None, block_size=None, return_both=False,
                 active_len=None, attention_mask=None):
      assert x_in.shape[-1] == 12  # concat L+L
      return torch.zeros(x_in.shape[0], 6, 8)

  m = SimpleNamespace(single_stream_train=False, backbone=_BB())
  xt = torch.randint(0, 8, (2, 6))
  x0 = torch.randint(0, 8, (2, 6))
  out = BlockTrainer._backbone_logits(m, xt, x0, block_size=4)
  assert out.shape == (2, 6, 8)
