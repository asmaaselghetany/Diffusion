"""Tier 0: BlockSampler shape checks with a tiny mock backbone."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from discrete_diffusion.sampling.block_sampler import BlockSampler


class _TinyBackbone(nn.Module):
  def __init__(self, n: int, vocab: int):
    super().__init__()
    self.n_tokens = n
    self.block_size = 8
    self.embed = nn.Embedding(vocab, 16)
    self.head = nn.Linear(16, vocab)

  def forward(self, indices, sigma=None, sample_mode=False, store_kv=False,
              **kwargs):
    del sigma, sample_mode, store_kv, kwargs
    h = self.embed(indices).mean(dim=1, keepdim=True).expand(-1, self.n_tokens, -1)
    return self.head(h)


class _MockBlockTrainer(nn.Module):
  def __init__(self, n=16, vocab=32, mode='masked'):
    super().__init__()
    self.num_tokens = n
    self.block_size = 8
    self.vocab_size = vocab
    self.mask_id = vocab - 1
    self.sampling_eps = 1e-3
    self.forward_process_name = mode
    self.backbone = _TinyBackbone(n, vocab)
    self.tokenizer = SimpleNamespace(
        bos_token_id=0, eos_token_id=2, pad_token_id=3)


    from discrete_diffusion.noise_schedules.log_linear import LogLinear
    self.noise = LogLinear(eps=1e-3)

  def backbone_logits(self, xt, x0, active_len=None, **kwargs):
    del x0, kwargs
    if active_len is not None:
      xt = xt[:, : int(active_len)]
    # Match real backbone: logits length == active span (sampler pads if needed).
    h = self.backbone.embed(xt).mean(dim=1, keepdim=True).expand(
        -1, xt.shape[1], -1)
    return self.backbone.head(h)

  def prior_sample(self, *batch_dims):
    size = batch_dims if len(batch_dims) > 1 else batch_dims[0]
    if self.forward_process_name == 'uniform':
      return torch.randint(0, self.vocab_size, size, dtype=torch.int64)
    return torch.full(size, self.mask_id, dtype=torch.int64)


def _config(mode: str):
  return SimpleNamespace(
      algo=SimpleNamespace(forward_process_name=mode),
      sampling=SimpleNamespace(
          steps=4, inject_bos=False, use_float64=False,
          use_arpc=False, arpc_mode='simplified',
          arpc_prefix_frac=0.25, arpc_resample_tau=0.5,
          arpc_corruption_mode='divergence',
          arpc_divergence_measure='kld',
          arpc_diffusion_metric='confidence',
          arpc_ar_metric='nll',
          arpc_warmup_steps=0, arpc_guide_every=1,
          arpc_temperature=1.0,
          arpc_use_prefix_fill=None,
          hierarchical_kv=False, use_block_cache=False,
          single_stream_decode=False,
          sub_block_size=None,
          align_shift_logits=None,
          pad_after_eos=True,
          stop_on_eos=True,
          greedy=False,
          p_nucleus=1.0,
          unmask_threshold=None,
      ),
  )


def test_block_sampler_masked_shape():
  model = _MockBlockTrainer(mode='masked')
  sampler = BlockSampler(_config('masked'))
  out = sampler.generate(model, num_samples=2, num_steps=4, eps=1e-3, inject_bos=False)
  assert out.shape == (2, 16)
  assert out.min() >= 0
  assert out.max() < model.vocab_size


def test_block_sampler_uniform_shape():
  model = _MockBlockTrainer(mode='uniform')
  sampler = BlockSampler(_config('uniform'))
  out = sampler.generate(model, num_samples=1, num_steps=4, eps=1e-3, inject_bos=False)
  assert out.shape == (1, 16)


def test_block_sampler_prefix_frozen():
  """Conditional gen keeps the prompt tokens unchanged (lm-eval path)."""
  model = _MockBlockTrainer(mode='masked', n=16)
  sampler = BlockSampler(_config('masked'))
  prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
  out = sampler.generate(
      model,
      num_samples=1,
      num_steps=4,
      eps=1e-3,
      inject_bos=False,
      prefix_ids=prefix,
  )
  assert out.shape == (1, 16)
  assert torch.equal(out[:, :4], prefix)


def test_pad_after_eos_keeps_first_eos_and_pads_rest():
  """Codex-fixes port: post-EOS tokens become pad in the sample tensor."""
  samples = torch.tensor([
      [9, 8, 2, 7, 2, 6],
      [9, 8, 7, 6, 5, 4],
  ])
  out = BlockSampler._pad_after_eos(
      samples.clone(), start=2, eos_id=2, pad_id=0)
  assert torch.equal(out[0], torch.tensor([9, 8, 2, 0, 0, 0]))
  assert torch.equal(out[1], torch.tensor([9, 8, 7, 6, 5, 4]))


def test_block_sampler_max_new_tokens_limits_blocks():
  model = _MockBlockTrainer(mode='masked', n=32)
  model.block_size = 8
  cfg = _config('masked')
  # Isolate the "don't denoise later blocks" check from pad-after-EOS rewrite.
  cfg.sampling.pad_after_eos = False
  sampler = BlockSampler(cfg)
  prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
  # Only enough for ~8 new tokens → should stop after completing block covering pos 4..12
  out = sampler.generate(
      model,
      num_samples=1,
      num_steps=2,
      eps=1e-3,
      inject_bos=False,
      prefix_ids=prefix,
      max_new_tokens=8,
  )
  assert out.shape == (1, 32)
  assert torch.equal(out[:, :4], prefix)
  # Later untouched prior region stays mask-filled for masked mode.
  assert (out[:, 16:] == model.mask_id).all()


def test_block_sampler_stop_on_eos_skips_later_blocks():
  model = _MockBlockTrainer(mode='masked', n=32)
  model.block_size = 8
  cfg = _config('masked')
  cfg.sampling.pad_after_eos = False
  cfg.sampling.stop_on_eos = True
  sampler = BlockSampler(cfg)

  # After first denoise window, force EOS so the block loop can early-stop.
  orig_denoise = sampler._denoise_block
  calls = {'n': 0}

  def denoise_with_early_eos(model_in, xt, x0, start, end, num_steps, eps,
                             inject_bos=True):
    xt, x0 = orig_denoise(
        model_in, xt, x0, start, end, num_steps, eps, inject_bos=inject_bos)
    calls['n'] += 1
    if calls['n'] == 1:
      xt = xt.clone()
      xt[:, start] = model_in.tokenizer.eos_token_id
      x0 = x0.clone()
      x0[:, start] = model_in.tokenizer.eos_token_id
    return xt, x0

  sampler._denoise_block = denoise_with_early_eos
  out = sampler.generate(
      model, num_samples=1, num_steps=2, eps=1e-3, inject_bos=False)
  assert calls['n'] == 1
  assert (out[:, 8:] == model.mask_id).all()


def test_shared_block_position_ids_duplicate_halves():
  from discrete_diffusion.models.qwen.modeling import shared_block_position_ids
  n, bsz = 8, 2
  ids = shared_block_position_ids(n, 'cpu', batch_size=bsz)
  assert ids.shape == (bsz, 2 * n)
  assert torch.equal(ids[0, :n], torch.arange(n))
  assert torch.equal(ids[0, n:], torch.arange(n))
  assert torch.equal(ids[0], ids[1])


def test_uniform_denoise_freezes_prefix_and_future():
  """Uniform reverse may redraw the full seq; only the active block is kept."""
  model = _MockBlockTrainer(mode='uniform')
  sampler = BlockSampler(_config('uniform'))
  n, bs = 16, 8
  start, end = bs, 2 * bs
  prefix = torch.arange(bs).unsqueeze(0).expand(2, -1)
  future = torch.full((2, n - end), 3)
  current = torch.full((2, bs), 5)
  xt = torch.cat([prefix, current, future], dim=-1)
  x0 = xt.clone()

  def scribble(_model, xt_in, _x0, _t, _dt, **kwargs):
    del _model, _x0, _t, _dt, kwargs
    return torch.full_like(xt_in, 7)

  sampler._uniform_step = scribble
  out, x0_out = sampler._denoise_block(
      model, xt.clone(), x0.clone(), start, end, num_steps=3, eps=1e-3)
  assert torch.equal(out[:, :start], prefix)
  assert torch.equal(x0_out[:, :start], prefix)
  assert (out[:, start:end] == 7).all()
  assert torch.equal(out[:, end:], future)
  assert torch.equal(x0_out[:, end:], future)


def test_masked_denoise_freezes_prefix_and_future():
  model = _MockBlockTrainer(mode='masked')
  sampler = BlockSampler(_config('masked'))
  n, bs = 16, 8
  start, end = bs, 2 * bs
  prefix = torch.arange(bs).unsqueeze(0).expand(2, -1)
  future = torch.full((2, n - end), model.mask_id)
  current = torch.full((2, bs), model.mask_id)
  xt = torch.cat([prefix, current, future], dim=-1)
  x0 = xt.clone()

  def scribble(_model, xt_in, _x0, _t, _dt, **kwargs):
    del _model, _x0, _t, _dt, kwargs
    return torch.full_like(xt_in, 7)

  sampler._masked_step = scribble
  out, _ = sampler._denoise_block(
      model, xt.clone(), x0.clone(), start, end, num_steps=2, eps=1e-3)
  assert torch.equal(out[:, :start], prefix)
  assert (out[:, start:end] == 7).all()
  assert torch.equal(out[:, end:], future)


def test_ignore_bos_frozen_in_block0():
  model = _MockBlockTrainer(mode='uniform')
  model.ignore_bos = True
  sampler = BlockSampler(_config('uniform'))
  n, bs = 16, 8
  xt = torch.randint(1, model.vocab_size, (1, n))
  xt[:, 0] = 0
  x0 = xt.clone()

  def scribble(_model, xt_in, _x0, _t, _dt, **kwargs):
    del _model, _x0, _t, _dt, kwargs
    return torch.full_like(xt_in, 7)

  sampler._uniform_step = scribble
  out, _ = sampler._denoise_block(
      model, xt.clone(), x0.clone(), 0, bs, num_steps=2, eps=1e-3)
  assert int(out[0, 0].item()) == 0
  assert (out[:, 1:bs] == 7).all()
