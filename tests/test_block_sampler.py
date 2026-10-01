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
          ar_block_bridge=None,
          ban_mask_pad_logits=True,
          pad_after_eos=True,
          stop_on_eos=True,
          greedy=False,
          p_nucleus=1.0,
          unmask_threshold=None,
          x0_temperature=1.0,
          posterior_sampler='fast',
          track_revisions=False,
          uniform_confidence_sticky=False,
          sticky_min_conf=0.0,
          uniform_commit_revise=False,
          uniform_commit_revise_tau=0.25,
          allow_full_seq_decode=False,
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


def test_ar_block_bridge_auto_on_for_masked_shift():
  cfg = _config('masked')
  sampler = BlockSampler(cfg)
  model = _MockBlockTrainer(mode='masked')
  model.shift_loss_targets = True
  assert sampler._ar_block_bridge_enabled(model) is True
  model.shift_loss_targets = False
  assert sampler._ar_block_bridge_enabled(model) is False
  cfg.sampling.ar_block_bridge = False
  sampler = BlockSampler(cfg)
  model.shift_loss_targets = True
  assert sampler._ar_block_bridge_enabled(model) is False


def test_ar_block_bridge_writes_next_mask_position():
  """Hub ~81-85: after a block, argmax at last committed pos seeds next."""
  model = _MockBlockTrainer(mode='masked', n=16, vocab=32)
  model.shift_loss_targets = True
  cfg = _config('masked')
  cfg.sampling.ar_block_bridge = True
  sampler = BlockSampler(cfg)
  end = 8
  xt = torch.randint(1, model.vocab_size - 1, (1, 16))
  xt[:, end:] = model.mask_id
  x0 = xt.clone()

  def fake_logits(xt_in, x0_in, active_len=None, **kwargs):
    del x0_in, kwargs
    a = int(active_len) if active_len is not None else xt_in.shape[1]
    out = torch.zeros(xt_in.shape[0], a, model.vocab_size)
    # Unshifted: position end-1 predicts token id 11.
    out[:, end - 1, 11] = 10.0
    return out

  model.backbone_logits = fake_logits
  xt2, x02 = sampler._ar_block_bridge(model, xt.clone(), x0.clone(), end)
  assert int(xt2[0, end].item()) == 11
  assert int(x02[0, end].item()) == 11
  assert torch.equal(xt2[:, :end], xt[:, :end])


def test_init_block_preserves_ar_bridge_seed():
  model = _MockBlockTrainer(mode='masked')
  sampler = BlockSampler(_config('masked'))
  n, bs = 16, 8
  xt = torch.full((1, n), model.mask_id, dtype=torch.long)
  x0 = xt.clone()
  # Simulate Hub AR seed at next-block start.
  xt[:, bs] = 7
  x0[:, bs] = 7
  xt2, x02 = sampler._init_block(model, xt, x0, bs, 2 * bs, inject_bos=False)
  assert int(xt2[0, bs].item()) == 7
  assert int(x02[0, bs].item()) == 7
  assert (xt2[:, bs + 1:2 * bs] == model.mask_id).all()


def test_dual_cache_refresh_on_mask_or_active_len():
  """Hub ~102: refresh when first small-block token is MASK / active_len mismatch."""
  model = _MockBlockTrainer(mode='masked')
  cfg = _config('masked')
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.use_block_cache = True
  cfg.sampling.single_stream_decode = True
  sampler = BlockSampler(cfg)
  cache = SimpleNamespace(active_len=16)
  sampler._dual_cache = cache
  xt = torch.full((1, 16), 5, dtype=torch.long)
  # First token unmasked + matching active_len → keep.
  sampler._maybe_refresh_dual_cache(
      model, xt, window_start=8, active_end=16)
  assert sampler._dual_cache is cache
  # First token MASK → refresh.
  xt[:, 8] = model.mask_id
  sampler._maybe_refresh_dual_cache(
      model, xt, window_start=8, active_end=16)
  assert sampler._dual_cache is None
  sampler._dual_cache = cache
  # active_len mismatch → refresh.
  xt[:, 8] = 5
  sampler._maybe_refresh_dual_cache(
      model, xt, window_start=8, active_end=8)
  assert sampler._dual_cache is None


def test_denoise_block_does_not_unconditionally_clear_dual_cache():
  """DualCache survives across ``_denoise_block`` when refresh does not fire."""
  model = _MockBlockTrainer(mode='masked')
  cfg = _config('masked')
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.use_block_cache = True
  cfg.sampling.single_stream_decode = True
  sampler = BlockSampler(cfg)
  sentinel = SimpleNamespace(active_len=16)
  sampler._dual_cache = sentinel
  n, bs = 16, 8
  start, end = bs, 2 * bs
  # Window start already unmasked so Hub refresh condition is false.
  xt = torch.randint(1, model.vocab_size - 1, (1, n))
  xt[:, end:] = model.mask_id
  x0 = xt.clone()

  def scribble(_model, xt_in, _x0, _t, _dt, **kwargs):
    del _model, _x0, _t, _dt, kwargs
    return xt_in

  sampler._masked_step = scribble
  sampler._denoise_block(
      model, xt, x0, start, end, num_steps=1, eps=1e-3, inject_bos=False)
  assert sampler._dual_cache is sentinel


def test_eos_stop_ready_requires_no_mask_before_eos():
  xt = torch.tensor([[1, 2, 3, 9, 5, 6]])  # eos=9
  assert BlockSampler._eos_stop_ready(
      xt, gen_start=1, end=6, eos_id=9, mask_id=0) is True
  xt2 = torch.tensor([[1, 2, 0, 9, 5, 6]])  # MASK before EOS
  assert BlockSampler._eos_stop_ready(
      xt2, gen_start=1, end=6, eos_id=9, mask_id=0) is False


def test_ban_mask_pad_logits_can_be_disabled():
  model = _MockBlockTrainer(mode='masked')
  cfg = _config('masked')
  cfg.sampling.ban_mask_pad_logits = False
  sampler = BlockSampler(cfg)
  logits = torch.zeros(1, 4, model.vocab_size)
  logits[..., model.mask_id] = 5.0
  out = sampler._prepare_masked_logits(model, logits.clone(), shift_mode='full')
  assert float(out[..., model.mask_id].max()) == 5.0


def test_uniform_rejects_greedy_argmax():
  """argmax(q_xs) locks prior noise on uniform; sampler must force ancestral."""
  model = _MockBlockTrainer(mode='uniform')
  sampler = BlockSampler(_config('uniform'))
  sampler.generate(
      model, num_samples=1, num_steps=2, eps=1e-3, inject_bos=False, greedy=True)
  assert sampler._greedy_decode is False

  model_m = _MockBlockTrainer(mode='masked')
  sampler_m = BlockSampler(_config('masked'))
  sampler_m.generate(
      model_m, num_samples=1, num_steps=2, eps=1e-3, inject_bos=False,
      greedy=True)
  assert sampler_m._greedy_decode is True


def test_uniform_auto_enables_truncated_packing():
  """Bare open-loop pins must not stay on full-seq dual."""
  cfg = _config('uniform')
  cfg.sampling.hierarchical_kv = False
  cfg.sampling.single_stream_decode = False
  sampler = BlockSampler(cfg)
  assert sampler.hierarchical_kv is True
  # Truncated dual is BlockGen-faithful; do not force Hub single-stream.
  assert sampler.single_stream_decode is False

  cfg2 = _config('uniform')
  cfg2.sampling.hierarchical_kv = True
  cfg2.sampling.single_stream_decode = False
  sampler2 = BlockSampler(cfg2)
  assert sampler2.hierarchical_kv is True
  assert sampler2.single_stream_decode is False


def test_masked_open_loop_auto_enables_truncation():
  """C0 ancestral 2.27% used full-seq dual; skeleton must force truncation."""
  cfg = _config('masked')
  cfg.sampling.hierarchical_kv = False
  cfg.sampling.single_stream_decode = False
  cfg.sampling.unmask_threshold = None
  cfg.sampling.use_block_cache = False
  sampler = BlockSampler(cfg)
  assert sampler.hierarchical_kv is True
  assert sampler.single_stream_decode is False
  # Confidence DualCache path must NOT be rewritten.
  cfg_dc = _config('masked')
  cfg_dc.sampling.hierarchical_kv = True
  cfg_dc.sampling.single_stream_decode = True
  cfg_dc.sampling.use_block_cache = True
  cfg_dc.sampling.unmask_threshold = 1.0
  sampler_dc = BlockSampler(cfg_dc)
  assert sampler_dc.use_block_cache is True
  assert sampler_dc.unmask_threshold == 1.0


def test_allow_full_seq_decode_escape_hatch():
  """Legacy ablation may keep full-seq dual; default must not."""
  cfg = _config('masked')
  cfg.sampling.hierarchical_kv = False
  cfg.sampling.single_stream_decode = False
  cfg.sampling.allow_full_seq_decode = True
  sampler = BlockSampler(cfg)
  assert sampler.hierarchical_kv is False
  assert sampler.single_stream_decode is False


def test_hierarchical_open_loop_uses_block_gen_packing():
  """Baseline hierarchical path calls BlockGen [prefix|block] packing."""
  cfg = _config('masked')
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.single_stream_decode = False
  cfg.sampling.use_block_cache = False
  sampler = BlockSampler(cfg)
  model = _MockBlockTrainer(n=16, vocab=32, mode='masked')
  calls = []

  def _block_gen(prefix, xt_b, block_size=None, **kwargs):
    del block_size
    calls.append((prefix.shape[-1], xt_b.shape[-1], kwargs.get('x0_causal')))
    b, l = xt_b.shape
    return torch.zeros(b, l, model.vocab_size, device=xt_b.device)

  model.backbone.block_gen_logits = _block_gen
  model.backbone.x0_causal = False
  xt = torch.full((2, 16), model.mask_id, dtype=torch.long)
  x0 = xt.clone()
  # Committed prefix of length 8.
  xt[:, :8] = 1
  x0[:, :8] = 1
  logits, mode = sampler._logits(model, xt, x0, active_end=16, window=(8, 16))
  assert mode == 'window'
  assert calls == [(8, 8, False)]
  assert logits.shape == (2, 16, 32)
  assert torch.count_nonzero(logits[:, :8]) == 0


def test_hierarchical_block_gen_forwards_x0_causal():
  """Hard BlockGen bake: trained x0_causal must reach packing decode."""
  cfg = _config('uniform')
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.single_stream_decode = False
  cfg.sampling.use_block_cache = False
  sampler = BlockSampler(cfg)
  model = _MockBlockTrainer(n=16, vocab=32, mode='uniform')
  seen = []

  def _block_gen(prefix, xt_b, block_size=None, **kwargs):
    del prefix, block_size
    seen.append(kwargs.get('x0_causal'))
    b, l = xt_b.shape
    return torch.zeros(b, l, model.vocab_size, device=xt_b.device)

  model.backbone.block_gen_logits = _block_gen
  model.backbone.x0_causal = True
  xt = torch.randint(0, 30, (2, 16))
  x0 = xt.clone()
  sampler._logits(model, xt, x0, active_end=16, window=(8, 16))
  assert seen == [True]


def test_uniform_truncated_active_end_does_not_ceil_into_unif_future():
  """Sub-block uniform must not attend undenoised Unif in the same attn block."""
  cfg = _config('uniform')
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.single_stream_decode = True
  cfg.sampling.sub_block_size = 8
  sampler = BlockSampler(cfg)
  model = SimpleNamespace(block_size=32)
  assert sampler._truncated_active_end(model, end=8, seq_len=128) == 8
  assert sampler._truncated_active_end(model, end=40, seq_len=128) == 40

  cfg_m = _config('masked')
  cfg_m.sampling.hierarchical_kv = True
  sampler_m = BlockSampler(cfg_m)
  assert sampler_m._truncated_active_end(model, end=40, seq_len=128) == 64


def test_uniform_posterior_argmax_locks_xt_at_mid_alpha():
  """Duo algebra: mid-α argmax(q) prefers noisy xt even when p(x0) is peaked.

  This is why FORCE_GREEDY=1 on U0 produces multilingual soup — not a
  mysterious train collapse. Final noise-removal (α_s=1) still recovers.
  """
  import torch.nn.functional as F

  v, b, l = 64, 4, 32
  torch.manual_seed(0)
  true = torch.randint(0, v, (b, l))
  xt = (true + 1) % v  # always wrong
  p = torch.full((b, l, v), 1e-8)
  p.scatter_(-1, true.unsqueeze(-1), 1.0)
  p = p / p.sum(dim=-1, keepdim=True)

  def argmax_q(alpha_t: float, *, noise_removal: bool):
    a_t = torch.full((b, l, 1), alpha_t)
    a_s = torch.ones_like(a_t) if noise_removal else torch.full(
        (b, l, 1), min(1.0, alpha_t + 0.2))
    a_t2, a_s2 = a_t.squeeze(-1), a_s.squeeze(-1)
    p_xt = torch.gather(p, -1, xt.unsqueeze(-1)).squeeze(-1)
    denom = (a_t2 * v * p_xt + (1.0 - a_t2)).clamp(min=1e-12)
    alpha_ts = a_t2 / a_s2.clamp(min=1e-8)
    xt_oh = F.one_hot(xt, v).float()
    uniform = torch.full((1, 1, v), 1.0 / v)
    num = (
        (a_t * v * p * xt_oh)
        + ((alpha_ts - a_t2).unsqueeze(-1) * xt_oh)
        + ((a_s - a_t) * p)
        + ((1 - alpha_ts) * (1 - a_s2)).unsqueeze(-1) * uniform)
    return (num / denom.unsqueeze(-1)).argmax(-1)

  mid = argmax_q(0.3, noise_removal=False)
  assert (mid == xt).float().mean() > 0.95, mid
  assert (mid == true).float().mean() < 0.05

  final = argmax_q(0.3, noise_removal=True)
  assert (final == true).float().mean() > 0.95, final


def test_uniform_sticky_freezes_max_conf_site():
  """DualCache analog: force-stick max-conf site; never re-Unif frozen."""
  model = _MockBlockTrainer(mode='uniform', n=8, vocab=16)
  cfg = _config('uniform')
  cfg.sampling.uniform_confidence_sticky = True
  cfg.sampling.unmask_threshold = 1.0
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.single_stream_decode = True
  sampler = BlockSampler(cfg)
  b, l, v = 2, 8, 16
  xt = torch.randint(0, v - 1, (b, l))
  p_x0 = torch.full((b, l, v), 1e-6)
  # Make position 3 overwhelmingly confident toward token 5.
  p_x0[:, 3, 5] = 1.0
  p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)
  xs0 = xt.clone()
  sampler._block_sticky_frozen = torch.zeros(b, l, dtype=torch.bool)
  out1 = sampler._apply_uniform_sticky(xs0, p_x0, xt, window=(0, l))
  assert (out1[:, 3] == 5).all()
  assert sampler._block_sticky_frozen[:, 3].all()
  # Non-commits keep Unif (xt) — Hub leaves MASK; ignore Duo scribble.
  other = [i for i in range(l) if i != 3]
  assert (out1[:, other] == xt[:, other]).all()
  # Second step: frozen site stays even if proposal tries to change it.
  xs_try = torch.full_like(xt, 7)
  out2 = sampler._apply_uniform_sticky(xs_try, p_x0, out1, window=(0, l))
  assert (out2[:, 3] == 5).all()


def test_uniform_sticky_ignores_duo_proposal():
  """Hub Twin: sticky must not write Duo samples onto undecided sites."""
  cfg = _config('uniform')
  cfg.sampling.uniform_confidence_sticky = True
  cfg.sampling.unmask_threshold = 1.0  # only force-max commits
  cfg.sampling.sticky_min_conf = 0.0
  sampler = BlockSampler(cfg)
  b, l, v = 1, 4, 8
  xt = torch.arange(l).unsqueeze(0)  # Unif state 0,1,2,3
  # Peak only at pos 1 → force-max commits token 7 there.
  p_x0 = torch.full((b, l, v), 1e-4)
  p_x0[:, 1, 7] = 1.0
  p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)
  duo_scribble = torch.full_like(xt, 5)  # would pollute if used
  sampler._block_sticky_frozen = torch.zeros(b, l, dtype=torch.bool)
  out = sampler._apply_uniform_sticky(duo_scribble, p_x0, xt, window=(0, l))
  assert int(out[0, 1]) == 7
  assert (out[0, [0, 2, 3]] == xt[0, [0, 2, 3]]).all()
  assert (out != 5).all()


def test_uniform_commit_revise_unfreezes_low_conf():
  """BlockGen-inspired revise: low token-conf commits get re-Unif'd."""
  model = _MockBlockTrainer(mode='uniform', n=8, vocab=16)
  cfg = _config('uniform')
  cfg.sampling.uniform_confidence_sticky = True
  cfg.sampling.uniform_commit_revise = True
  cfg.sampling.uniform_commit_revise_tau = 0.9
  cfg.sampling.unmask_threshold = 0.9
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.single_stream_decode = True
  sampler = BlockSampler(cfg)
  b, l, v = 1, 8, 16
  xt = torch.zeros(b, l, dtype=torch.long)
  x0 = xt.clone()
  sampler._block_sticky_frozen = torch.ones(b, l, dtype=torch.bool)
  # Mock logits path: make all token confidences tiny so revise fires.
  real_logits = sampler._logits

  def _weak(model, xt_in, x0_in, **kwargs):
    logits, mode = real_logits(model, xt_in, x0_in, **kwargs)
    # Flat distribution → gather conf ≈ 1/V << 0.9
    return torch.zeros_like(logits), mode

  sampler._logits = _weak
  n = sampler._uniform_commit_revise_low_conf(
      model, xt, x0, 0, l, active_end=l)
  assert n == l
  assert not sampler._block_sticky_frozen.any()


def test_uniform_sticky_denoise_uses_confidence_until_not_ancestral():
  """uniform_dual must not fall through to mid-α Unif ancestral.

  Mirror masked DualCache: noise-removal (dt=None) until the window is
  frozen. Regression for the ~34% flex sticky schedule footgun.
  """
  model = _MockBlockTrainer(mode='uniform', n=16, vocab=32)
  model.block_size = 8
  cfg = _config('uniform')
  cfg.sampling.uniform_confidence_sticky = True
  cfg.sampling.unmask_threshold = 1.0
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.single_stream_decode = True
  cfg.sampling.steps = 4  # ancestral would only run 4 mid-α steps
  sampler = BlockSampler(cfg)

  real_uniform = sampler._uniform_step
  calls = {'n': 0, 'none_dt': 0}

  def _counting(model, xt, x0, t_scalar, dt, **kwargs):
    calls['n'] += 1
    if dt is None:
      calls['none_dt'] += 1
    return real_uniform(model, xt, x0, t_scalar, dt, **kwargs)

  sampler._uniform_step = _counting
  xt = torch.randint(0, 31, (1, 16))
  x0 = xt.clone()
  out_xt, _ = sampler._denoise_block(
      model, xt, x0, start=0, end=8, num_steps=4, eps=1e-3, inject_bos=False)
  # Window must be fully frozen after confidence-until.
  assert sampler._block_sticky_frozen[:, 0:8].all()
  # Every reverse step was noise-removal (dt=None), not mid-α ancestral.
  assert calls['n'] >= 8  # force-max ≥1 commit/step → ≥8 passes
  assert calls['none_dt'] == calls['n']
  assert out_xt.shape == (1, 16)


def test_ucc_survives_sticky_min_gate_and_batch():
  """Regression for bake-U0-UC: exhausted bound / early flush return.

  Flat p_x0 ⇒ peak ≪ sticky_min_conf so thr/force-max stall every step.
  Budget must be batch-aware and flush must continue until the window
  is fully frozen (old code used end-start and returned after one flush).
  """
  model = _MockBlockTrainer(mode='uniform', n=16, vocab=64)
  model.block_size = 8
  cfg = _config('uniform')
  cfg.sampling.uniform_confidence_sticky = True
  cfg.sampling.unmask_threshold = 0.9
  cfg.sampling.sticky_min_conf = 0.5
  cfg.sampling.hierarchical_kv = True
  cfg.sampling.single_stream_decode = True
  cfg.sampling.uniform_commit_revise = False
  sampler = BlockSampler(cfg)

  real_logits = sampler._logits

  def _flat(model, xt_in, x0_in, **kwargs):
    logits, mode = real_logits(model, xt_in, x0_in, **kwargs)
    return torch.zeros_like(logits), mode

  sampler._logits = _flat
  b = 4
  xt = torch.randint(0, 63, (b, 16))
  x0 = xt.clone()
  out_xt, _ = sampler._denoise_block(
      model, xt, x0, start=0, end=8, num_steps=4, eps=1e-3, inject_bos=False)
  assert sampler._block_sticky_frozen is not None
  assert sampler._block_sticky_frozen[:, 0:8].all()
  assert out_xt.shape == (b, 16)
