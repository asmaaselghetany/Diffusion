"""ARPC scoring + hierarchical / blockgen sampler smoke tests."""

from types import SimpleNamespace

import torch

from discrete_diffusion.sampling.arpc import (
    causal_log_probs_for_span,
    corruption_indices,
    diffusion_scores,
    divergence_scores,
)
from discrete_diffusion.sampling.block_sampler import BlockSampler
from test_block_sampler import _MockBlockTrainer, _config


def test_divergence_scores_shapes():
  b, l, v = 2, 4, 8
  p_ar = torch.softmax(torch.randn(b, l, v), dim=-1)
  p_diff = torch.softmax(torch.randn(b, l, v), dim=-1)
  for m in ('kld', 'reverse_kld', 'tvd'):
    s = divergence_scores(p_ar, p_diff, measure=m)
    assert s.shape == (b, l)


def test_corruption_indices_excludes_prefix():
  torch.manual_seed(0)
  b, l, v = 2, 8, 16
  log_p = torch.randn(b, l, v).log_softmax(-1)
  x = torch.randint(0, v, (b, l))
  idxs = corruption_indices(
      log_p_x0=log_p,
      log_p_ar=log_p,
      x_current=x,
      num_to_corrupt=3,
      block_prefix_len=2,
      corruption_mode='random',
  )
  assert idxs.shape == (b, 3)
  assert (idxs >= 2).all()


def test_diffusion_and_causal_helpers():
  log_p = torch.randn(2, 4, 10).log_softmax(-1)
  s = diffusion_scores(log_p, metric='confidence')
  assert s.shape == (2, 4)
  logits = torch.randn(2, 6, 10)
  out = causal_log_probs_for_span(logits, start=2, end=6, vocab_size=10)
  assert out.shape == (2, 4, 10)


def test_blockgen_arpc_uniform_generate():
  model = _MockBlockTrainer(mode='uniform', n=16)
  model.backbone.causal_logits = lambda ids: torch.randn(
      ids.shape[0], ids.shape[1], model.vocab_size)
  cfg = _config('uniform')
  cfg.sampling.use_arpc = True
  cfg.sampling.arpc_mode = 'blockgen'
  cfg.sampling.arpc_corruption_mode = 'ar_metric'
  cfg.sampling.arpc_ar_metric = 'nll'
  cfg.sampling.arpc_use_prefix_fill = False
  cfg.sampling.arpc_warmup_steps = 0
  cfg.sampling.arpc_guide_every = 1
  cfg.sampling.hierarchical_kv = False
  sampler = BlockSampler(cfg)
  out = sampler.generate(
      model, num_samples=1, num_steps=4, eps=1e-3, inject_bos=False)
  assert out.shape == (1, 16)


def test_blockgen_arpc_absorb_remasks_lowest_ll():
  """Masked BlockGen ARPC: propose → AR NLL score → remask top-k with MASK."""
  model = _MockBlockTrainer(mode='masked', n=16)
  model.block_size = 8
  # Deterministic AR: token 3 is high-LL; others look bad under NLL scoring.
  def _causal(ids):
    logits = torch.full(
        (ids.shape[0], ids.shape[1], model.vocab_size), -10.0)
    logits[:, :, 3] = 10.0
    return logits
  model.backbone.causal_logits = _causal
  cfg = _config('masked')
  cfg.sampling.use_arpc = True
  cfg.sampling.arpc_mode = 'blockgen'
  cfg.sampling.arpc_corruption_mode = 'ar_metric'
  cfg.sampling.arpc_ar_metric = 'nll'
  cfg.sampling.arpc_use_prefix_fill = False
  cfg.sampling.arpc_warmup_steps = 0
  cfg.sampling.arpc_guide_every = 1
  cfg.sampling.unmask_threshold = None
  sampler = BlockSampler(cfg)
  out = sampler.generate(
      model, num_samples=1, num_steps=4, eps=1e-3, inject_bos=False)
  assert out.shape == (1, 16)
  # Guided steps remask; final noise-removal may leave some MASK if incomplete.
  # Smoke: path runs without refuse and stays in vocab.
  assert out.min() >= 0
  assert out.max() < model.vocab_size


def test_hierarchical_kv_truncated_path():
  model = _MockBlockTrainer(mode='masked', n=16)
  cfg = _config('masked')
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.sub_block_size = 4
  sampler = BlockSampler(cfg)
  out = sampler.generate(
      model, num_samples=1, num_steps=2, eps=1e-3, inject_bos=False)
  assert out.shape == (1, 16)


def test_arpc_on_hybrid_allowed():
  cfg = _config('hybrid')
  cfg.algo.hybrid_decode = 'masked'
  cfg.sampling.use_arpc = True
  cfg.sampling.arpc_mode = 'blockgen'
  sampler = BlockSampler(cfg)
  assert sampler.use_arpc
  assert sampler.mode == 'masked'


def test_masked_arpc_refuses_confidence_unmask():
  model = _MockBlockTrainer(mode='masked', n=16)
  model.backbone.causal_logits = lambda ids: torch.randn(
      ids.shape[0], ids.shape[1], model.vocab_size)
  cfg = _config('masked')
  cfg.sampling.use_arpc = True
  cfg.sampling.arpc_mode = 'blockgen'
  cfg.sampling.unmask_threshold = 0.9
  sampler = BlockSampler(cfg)
  try:
    sampler.generate(
        model, num_samples=1, num_steps=2, eps=1e-3, inject_bos=False)
    raised = False
  except ValueError as e:
    raised = True
    assert 'unmask_threshold' in str(e)
  assert raised


def test_absorb_arpc_guided_step_writes_mask():
  """Corrector must remask (not Unif-redraw) on the absorb arm."""
  torch.manual_seed(0)
  model = _MockBlockTrainer(mode='masked', n=16)
  model.block_size = 8

  def _causal(ids):
    logits = torch.full(
        (ids.shape[0], ids.shape[1], model.vocab_size), -5.0)
    logits[:, :, 3] = 5.0
    return logits

  model.backbone.causal_logits = _causal
  cfg = _config('masked')
  cfg.sampling.use_arpc = True
  cfg.sampling.arpc_mode = 'blockgen'
  cfg.sampling.arpc_corruption_mode = 'ar_metric'
  cfg.sampling.arpc_ar_metric = 'nll'
  cfg.sampling.arpc_use_prefix_fill = False
  sampler = BlockSampler(cfg)
  xt = torch.randint(0, model.vocab_size - 1, (1, 16))
  x0 = xt.clone()
  # Force a fully clean proposal block, then guide once.
  start, end = 0, 8
  t = torch.ones(1)
  sampler._arpc_guided_step(
      model, xt, x0, start, end, t, dt=0.25, block_prefix_len=0)
  remasked = (xt[:, start:end] == model.mask_id).sum().item()
  assert remasked >= 1, 'absorb ARPC must remask ≥1 lowest-LL token'


def test_uniform_arpc_guided_step_redraws_unif_not_mask():
  """Corrector must Unif-redraw (never plant MASK) on the uniform arm."""
  torch.manual_seed(0)
  model = _MockBlockTrainer(mode='uniform', n=16)
  model.block_size = 8

  def _causal(ids):
    logits = torch.full(
        (ids.shape[0], ids.shape[1], model.vocab_size), -5.0)
    logits[:, :, 3] = 5.0
    return logits

  model.backbone.causal_logits = _causal
  cfg = _config('uniform')
  cfg.sampling.use_arpc = True
  cfg.sampling.arpc_mode = 'blockgen'
  cfg.sampling.arpc_corruption_mode = 'ar_metric'
  cfg.sampling.arpc_ar_metric = 'nll'
  cfg.sampling.arpc_use_prefix_fill = False
  sampler = BlockSampler(cfg)
  # Avoid MASK in the clean proposal so any MASK after guide is from corruptor.
  content_v = model.vocab_size - 1  # mask_id is last id on the mock
  xt = torch.randint(0, content_v, (1, 16))
  x0 = xt.clone()
  before = xt[:, :8].clone()
  start, end = 0, 8
  t = torch.ones(1)
  sampler._arpc_guided_step(
      model, xt, x0, start, end, t, dt=0.25, block_prefix_len=0)
  block = xt[:, start:end]
  assert (block == model.mask_id).sum().item() == 0, (
      'uniform ARPC must not write MASK; expected Unif redraw')
  assert not torch.equal(block, before), (
      'uniform ARPC corrector must change ≥1 site via Unif redraw')


def test_simplified_arpc_still_works():
  model = _MockBlockTrainer(mode='uniform', n=16)
  model.backbone.causal_logits = lambda ids: torch.zeros(
      ids.shape[0], ids.shape[1], model.vocab_size)
  cfg = _config('uniform')
  cfg.sampling.use_arpc = True
  cfg.sampling.arpc_mode = 'simplified'
  cfg.sampling.arpc_prefix_frac = 0.25
  cfg.sampling.arpc_resample_tau = 0.01
  sampler = BlockSampler(cfg)
  out = sampler.generate(
      model, num_samples=1, num_steps=3, eps=1e-3, inject_bos=False)
  assert out.shape == (1, 16)


def test_simplified_arpc_prefix_survives_denoise():
  """AR-filled prefix tokens must not be redrawn by uniform reverse steps."""
  model = _MockBlockTrainer(mode='uniform', n=16)
  model.block_size = 8
  # Deterministic AR: always emit token 3.
  def _causal(ids):
    logits = torch.zeros(ids.shape[0], ids.shape[1], model.vocab_size)
    logits[:, :, 3] = 10.0
    return logits
  model.backbone.causal_logits = _causal
  cfg = _config('uniform')
  cfg.sampling.use_arpc = True
  cfg.sampling.arpc_mode = 'simplified'
  cfg.sampling.arpc_prefix_frac = 0.5  # 4 of 8
  cfg.sampling.arpc_resample_tau = 0.0  # disable post corrector
  cfg.sampling.arpc_use_prefix_fill = True
  sampler = BlockSampler(cfg)
  xt = model.prior_sample(1, 16)
  x0 = xt.clone()
  start, end = 0, 8
  xt, x0 = sampler._init_block(model, xt, x0, start, end)
  # Positions 1..3 should be AR-filled with 3 (pos0 may be bos/prior).
  filled = xt[:, 1:4].clone()
  assert (filled == 3).all()
  xt2, _ = sampler._denoise_block(
      model, xt.clone(), x0.clone(), start, end, num_steps=3, eps=1e-3)
  assert torch.equal(xt2[:, 1:4], filled)
