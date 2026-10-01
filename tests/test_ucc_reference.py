"""Independent slow UCC reference vs BlockSampler fast path (Part-1 test 7).

The reference reimplements only the *commit loop* (thr + force-max + freeze),
calling the same ``_logits`` stack for proposals so a logic bug in
``_apply_uniform_sticky`` / ``_ucc_until_frozen`` shows up as a token mismatch.

Also asserts Unif sub-window packing: ``active_end == window_end`` so later
sub-windows' prior-sample Unif never enters the forward (Hub MASK may densify
the full attention block; Unif must not).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from discrete_diffusion.sampling.block_sampler import BlockSampler


class _DetBackbone(nn.Module):
  """Position- and content-dependent logits (deterministic, full vocab peaks)."""

  def __init__(self, vocab: int, block_size: int = 8):
    super().__init__()
    self.block_size = block_size
    self.vocab = vocab
    self.seen_spans: list[tuple[int, int]] = []  # (active_len, window-ish)

  def block_eval_logits(self, xt, *, active_len=None, block_size=None, **kw):
    del kw, block_size
    a = int(active_len) if active_len is not None else xt.shape[1]
    self.seen_spans.append((a, int(xt.shape[1])))
    ids = xt[:, :a]
    b, l = xt.shape
    logits = torch.full((b, l, self.vocab), -20.0)
    for i in range(a):
      # Peak vocabulary index depends on position + visible prefix checksum.
      chk = int(ids[0, : i + 1].sum().item()) if b else 0
      peak = (chk * 17 + i * 3) % (self.vocab - 1)
      logits[:, i, peak] = 10.0
      # Slight mass on current token so thr=1 stays inert.
      cur = int(ids[0, i].item()) % self.vocab
      logits[:, i, cur] = max(float(logits[0, i, cur]), 0.5)
    return logits


class _DetTrainer(nn.Module):
  def __init__(self, n=32, vocab=64, mode='uniform', block_size=8):
    super().__init__()
    self.num_tokens = n
    self.block_size = block_size
    self.vocab_size = vocab
    self.mask_id = vocab - 1
    self.sampling_eps = 1e-3
    self.forward_process_name = mode
    self.backbone = _DetBackbone(vocab, block_size=block_size)
    self.tokenizer = SimpleNamespace(
        bos_token_id=0, eos_token_id=2, pad_token_id=3)
    from discrete_diffusion.noise_schedules.log_linear import LogLinear
    self.noise = LogLinear(eps=1e-3)

  def backbone_logits(self, xt, x0, active_len=None, **kwargs):
    del x0, kwargs
    return self.backbone.block_eval_logits(xt, active_len=active_len)

  def prior_sample(self, *batch_dims):
    size = batch_dims if len(batch_dims) > 1 else batch_dims[0]
    # Fixed pattern (not true RNG) so slow/fast share the same prior draw when
    # we seed via clone of a shared tensor instead.
    if self.forward_process_name == 'uniform':
      return torch.arange(self.vocab_size - 1).repeat(1024)[: size[-1]].unsqueeze(
          0).expand(*size[:-1], size[-1]).contiguous() if isinstance(
              size, tuple) else torch.arange(
                  self.vocab_size - 1).repeat(1024)[:size[1]].unsqueeze(0)
    return torch.full(size, self.mask_id, dtype=torch.int64)


def _cfg(**kw):
  base = dict(
      steps=8, inject_bos=False, use_float64=False,
      use_arpc=False, arpc_mode='simplified',
      arpc_prefix_frac=0.25, arpc_resample_tau=0.5,
      arpc_corruption_mode='divergence',
      arpc_divergence_measure='kld',
      arpc_diffusion_metric='confidence',
      arpc_ar_metric='nll',
      arpc_warmup_steps=0, arpc_guide_every=1,
      arpc_temperature=1.0, arpc_use_prefix_fill=None,
      hierarchical_kv=True, use_block_cache=False,
      single_stream_decode=True, sub_block_size=8,
      align_shift_logits=None, ar_block_bridge=None,
      ban_mask_pad_logits=True, pad_after_eos=False, stop_on_eos=False,
      greedy=True, p_nucleus=1.0, unmask_threshold=1.0,
      x0_temperature=1.0, posterior_sampler='fast', track_revisions=False,
      uniform_confidence_sticky=True, sticky_min_conf=0.0,
      uniform_commit_revise=False, uniform_commit_revise_tau=0.25,
      uniform_commit_random=False, uniform_commit_order='confidence',
      allow_full_seq_decode=False,
  )
  base.update(kw)
  return SimpleNamespace(
      algo=SimpleNamespace(forward_process_name='uniform'),
      sampling=SimpleNamespace(**base),
  )


def slow_ucc_commit_window(
    *,
    p_x0_fn,
    xt: torch.Tensor,
    start: int,
    end: int,
    thr: float,
    order: str = 'confidence',
    sticky_min_conf: float = 0.0,
    max_iters: int | None = None,
) -> torch.Tensor:
  """Dumb loop-by-loop UCC on one window. Independent of BlockSampler sticky.

  ``p_x0_fn(xt) -> [B, L, V]`` probabilities (already tempered / nucleus).
  Commits argmax tokens; non-commits keep current ``xt``.
  """
  out = xt.clone()
  b, _ = out.shape
  frozen = torch.zeros(b, out.shape[1], dtype=torch.bool, device=out.device)
  if start > 0:
    frozen[:, :start] = True
  budget = max_iters or max((end - start) * b * 2, end - start, 1)

  def _unfrozen() -> int:
    return int((~frozen[:, start:end]).sum().item())

  def _step(force_min_conf: float) -> None:
    nonlocal out, frozen
    p = p_x0_fn(out)
    conf = p.max(dim=-1).values
    hard = p.argmax(dim=-1)
    cand = torch.zeros_like(frozen)
    cand[:, start:end] = True
    cand = cand & ~frozen
    if order == 'ltr':
      pos = torch.arange(out.shape[1], device=out.device, dtype=conf.dtype)
      rank = -pos.unsqueeze(0).expand_as(conf)
    elif order == 'random':
      raise ValueError('slow reference: use confidence or ltr for determinism')
    else:
      rank = conf
    thr_stick = cand & (conf >= float(thr))
    rank_c = rank.masked_fill(~cand, float('-inf'))
    max_idx = rank_c.argmax(dim=-1)
    rows = torch.arange(b, device=out.device)
    still = cand.any(dim=-1)
    force = torch.zeros_like(cand)
    if still.any():
      peak = conf[rows[still], max_idx[still]]
      allow = peak >= float(force_min_conf)
      if allow.any():
        force[rows[still][allow], max_idx[still][allow]] = True
    new_stick = (thr_stick | force) & cand
    out = torch.where(new_stick, hard, out)
    frozen = frozen | new_stick

  for _ in range(budget):
    if bool(frozen[:, start:end].all()):
      break
    before = _unfrozen()
    _step(sticky_min_conf)
    if _unfrozen() >= before:
      if sticky_min_conf > 0:
        _step(0.0)
        continue
      raise RuntimeError('slow UCC stalled')
  # Final flush like the fast path.
  for _ in range(max(_unfrozen() + 2, 1)):
    if bool(frozen[:, start:end].all()):
      break
    _step(0.0)
  if not bool(frozen[:, start:end].all()):
    raise RuntimeError('slow UCC exhausted bound')
  return out


def _p_x0_from_sampler(sampler: BlockSampler, model, xt, x0, start, end):
  active_end = sampler._truncated_active_end(model, end, xt.shape[1])


  def fn(cur_xt):
    # Keep x0 in sync with commits (matches _apply in _denoise_block).
    cur_x0 = cur_xt.clone()
    raw, shift_mode = sampler._logits(
        model, cur_xt, cur_x0, active_end=active_end, window=(start, end))
    logits = sampler._prepare_masked_logits(
        model, raw, window=(start, end), shift_mode=shift_mode)
    if logits.size(1) < cur_xt.shape[1]:
      pad = torch.zeros(
          cur_xt.shape[0], cur_xt.shape[1] - logits.size(1), logits.size(-1),
          device=logits.device, dtype=logits.dtype)
      logits = torch.cat([logits, pad], dim=1)
    p = F.log_softmax(logits, dim=-1).exp()
    return sampler._apply_x0_temperature(p)

  return fn, active_end


def test_uniform_subwindow_active_end_is_window_end():
  """Unif must not densify to attention-block end (would see later Unif noise)."""
  model = _DetTrainer(n=32, block_size=8)
  sampler = BlockSampler(_cfg(sub_block_size=8))
  # Sub-window inside block 0..8 would be whole; use block 1 with sub-windows.
  # Window [8,16) inside attention block [8,16) if bs=8 — use n=32 bs=16.
  model.block_size = 16
  model.backbone.block_size = 16
  sampler = BlockSampler(_cfg(sub_block_size=8))
  spans = []

  def wrap_logits(m, xt, x0, *, active_end=None, window=None):
    spans.append((active_end, window))
    return orig(m, xt, x0, active_end=active_end, window=window)

  orig = sampler._logits
  sampler._logits = wrap_logits  # type: ignore[method-assign]
  prefix = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
  sampler.generate(
      model, num_samples=1, num_steps=8, eps=1e-3, inject_bos=False,
      prefix_ids=prefix, max_new_tokens=16, greedy=True)
  assert spans, 'no _logits calls'
  for active_end, window in spans:
    assert window is not None
    w0, w1 = window
    assert active_end == w1, (
        f'Unif active_end={active_end} != window_end={w1}; '
        f'later sub-window Unif would be visible (window={window})')


def test_slow_reference_matches_fast_ucc_confidence():
  model = _DetTrainer(n=24, vocab=48, block_size=8)
  sampler = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='confidence', sub_block_size=None))
  start, end = 8, 16
  # Shared initial state: committed prefix + Unif window + Unif future.
  prefix = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
  window = torch.tensor(
      [[3, 7, 11, 15, 19, 23, 27, 31]], dtype=torch.long)
  future = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]], dtype=torch.long)
  xt0 = torch.cat([prefix, window, future], dim=-1)
  x00 = xt0.clone()

  # Fast path.
  fast_xt, _ = sampler._denoise_block(
      model, xt0.clone(), x00.clone(), start, end, num_steps=8, eps=1e-3)

  # Slow reference on the same init (fresh sampler so sticky state is clean).
  sampler2 = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='confidence', sub_block_size=None))
  p_fn, active_end = _p_x0_from_sampler(
      sampler2, model, xt0.clone(), x00.clone(), start, end)
  assert active_end == end
  slow_xt = slow_ucc_commit_window(
      p_x0_fn=p_fn,
      xt=xt0.clone(),
      start=start,
      end=end,
      thr=1.0,
      order='confidence',
      sticky_min_conf=0.0,
  )
  assert torch.equal(fast_xt[:, start:end], slow_xt[:, start:end]), (
      f'fast={fast_xt[0, start:end].tolist()} '
      f'slow={slow_xt[0, start:end].tolist()}')
  assert torch.equal(fast_xt[:, :start], prefix)
  assert torch.equal(fast_xt[:, end:], future)


def test_slow_reference_matches_fast_ucc_ltr():
  model = _DetTrainer(n=24, vocab=48, block_size=8)
  sampler = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='ltr', sub_block_size=None))
  start, end = 8, 16
  prefix = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
  window = torch.tensor(
      [[5, 9, 13, 17, 21, 25, 29, 33]], dtype=torch.long) % 47
  future = torch.full((1, 8), 4)
  xt0 = torch.cat([prefix, window, future], dim=-1)

  fast_xt, _ = sampler._denoise_block(
      model, xt0.clone(), xt0.clone(), start, end, num_steps=8, eps=1e-3)

  sampler2 = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='ltr', sub_block_size=None))
  p_fn, _ = _p_x0_from_sampler(
      sampler2, model, xt0.clone(), xt0.clone(), start, end)
  slow_xt = slow_ucc_commit_window(
      p_x0_fn=p_fn, xt=xt0.clone(), start=start, end=end,
      thr=1.0, order='ltr', sticky_min_conf=0.0)
  assert torch.equal(fast_xt[:, start:end], slow_xt[:, start:end])


def test_force_max_tie_breaks_leftmost():
  """Equal peak conf on several sites → torch.argmax picks lowest index.

  Both fast and slow must commit the same leftmost tied site each step.
  """
  model = _DetTrainer(n=16, vocab=32, block_size=8)

  class _TieBackbone(nn.Module):
    def __init__(self, vocab, block_size):
      super().__init__()
      self.block_size = block_size
      self.vocab = vocab

    def block_eval_logits(self, xt, *, active_len=None, block_size=None, **kw):
      del kw, block_size
      a = int(active_len) if active_len is not None else xt.shape[1]
      b, l = xt.shape
      logits = torch.full((b, l, self.vocab), -50.0)
      # Identical one-hot-ish peak at vocab=1 for every position in the window
      # → equal conf; force-max must break ties left-to-right via argmax.
      for i in range(a):
        logits[:, i, 1] = 5.0
      return logits

  model.backbone = _TieBackbone(model.vocab_size, model.block_size)
  start, end = 0, 8
  xt0 = torch.full((1, 16), 3, dtype=torch.long)

  sampler = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='confidence', sub_block_size=None))
  fast_xt, _ = sampler._denoise_block(
      model, xt0.clone(), xt0.clone(), start, end, num_steps=8, eps=1e-3)

  sampler2 = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='confidence', sub_block_size=None))
  p_fn, _ = _p_x0_from_sampler(
      sampler2, model, xt0.clone(), xt0.clone(), start, end)
  slow_xt = slow_ucc_commit_window(
      p_x0_fn=p_fn, xt=xt0.clone(), start=start, end=end,
      thr=1.0, order='confidence', sticky_min_conf=0.0)

  assert torch.equal(fast_xt[:, start:end], slow_xt[:, start:end])
  # All committed to token 1 (the tied peak).
  assert (fast_xt[:, start:end] == 1).all()
  # With equal conf, thr=1 never fires (softmax peak < 1); order is L→R
  # via argmax-on-ties. Check left-fill trajectory isn't scrambled:
  # after full freeze every site is 1 — order verified by slow/fast match.


def test_thr1_edge_force_max_dominated_and_exact_one():
  """thr=1.0: diffuse p_x0 → only force-max; one-hot site → thr commit."""
  model = _DetTrainer(n=16, vocab=32, block_size=8)
  start, end = 0, 8

  # --- diffuse: thr share must be ~0 ---
  class _Diffuse(nn.Module):
    def __init__(self, vocab, block_size):
      super().__init__()
      self.block_size = block_size
      self.vocab = vocab

    def block_eval_logits(self, xt, *, active_len=None, block_size=None, **kw):
      del kw, block_size, xt
      a = int(active_len) if active_len is not None else 16
      # Flat logits → uniform-ish softmax; max conf ≪ 1.
      return torch.zeros(1, 16, self.vocab)

  model.backbone = _Diffuse(model.vocab_size, model.block_size)
  sampler = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='confidence', sub_block_size=None))
  sampler._reset_nfe_stats()
  xt0 = torch.full((1, 16), 4, dtype=torch.long)
  sampler._denoise_block(
      model, xt0.clone(), xt0.clone(), start, end, num_steps=8, eps=1e-3)
  n_thr = int(sampler._nfe_stats.get('n_thr_commits', 0))
  n_force = int(sampler._nfe_stats.get('n_force_max_commits', 0))
  assert n_thr == 0, f'expected no thr commits at thr=1 on flat logits, got {n_thr}'
  assert n_force >= (end - start), f'force-max under-committed: {n_force}'

  # --- exact conf=1 at one site: thr must fire there ---
  class _OneHot(nn.Module):
    def __init__(self, vocab, block_size):
      super().__init__()
      self.block_size = block_size
      self.vocab = vocab

    def block_eval_logits(self, xt, *, active_len=None, block_size=None, **kw):
      del kw, block_size
      logits = torch.full((1, 16, self.vocab), -80.0)
      # Position 3: exact one-hot on token 7 → conf=1 after softmax.
      logits[0, 3, 7] = 80.0
      # Other positions: mild peak so force-max can finish the window.
      for i in range(16):
        if i == 3:
          continue
        logits[0, i, (i + 2) % (self.vocab - 1)] = 2.0
      return logits

  model.backbone = _OneHot(model.vocab_size, model.block_size)
  sampler = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='confidence', sub_block_size=None))
  sampler._reset_nfe_stats()
  xt1 = torch.full((1, 16), 4, dtype=torch.long)
  out, _ = sampler._denoise_block(
      model, xt1.clone(), xt1.clone(), start, end, num_steps=8, eps=1e-3)
  n_thr = int(sampler._nfe_stats.get('n_thr_commits', 0))
  assert n_thr >= 1, 'exact conf=1 site should thr-commit at thr=1.0'
  assert int(out[0, 3].item()) == 7

  # Slow path agrees on the one-hot case.
  sampler2 = BlockSampler(_cfg(
      unmask_threshold=1.0, sticky_min_conf=0.0,
      uniform_commit_order='confidence', sub_block_size=None))
  p_fn, _ = _p_x0_from_sampler(
      sampler2, model, xt1.clone(), xt1.clone(), start, end)
  slow = slow_ucc_commit_window(
      p_x0_fn=p_fn, xt=xt1.clone(), start=start, end=end,
      thr=1.0, order='confidence', sticky_min_conf=0.0)
  assert torch.equal(out[:, start:end], slow[:, start:end])


def test_next_subwindow_forward_excludes_later_unif():
  """When window advances 8→16, logits must not see tokens in [16, 24)."""
  model = _DetTrainer(n=32, vocab=48, block_size=16)
  sampler = BlockSampler(_cfg(sub_block_size=8, unmask_threshold=1.0))
  seen_beyond: list[str] = []

  orig_eval = model.backbone.block_eval_logits

  def probe(xt, *, active_len=None, block_size=None, **kw):
    a = int(active_len) if active_len is not None else xt.shape[1]
    # Poison tokens beyond active_len; if the impl sliced wrong and read them,
    # peaks would move — we also assert the API only receives truncated use.
    if a < xt.shape[1]:
      # Caller passed full xt; real Hub path slices inside. Ensure our probe
      # only uses :a (same contract as modeling.block_eval_logits).
      poisoned = xt.clone()
      poisoned[:, a:] = 99  # invalid marker; must not affect :a logits
      out = orig_eval(poisoned, active_len=a, block_size=block_size, **kw)
      # Compare to clean
      clean = orig_eval(xt, active_len=a, block_size=block_size, **kw)
      if not torch.equal(out[:, :a], clean[:, :a]):
        seen_beyond.append(f'active_len={a}')
      return clean
    return orig_eval(xt, active_len=active_len, block_size=block_size, **kw)

  model.backbone.block_eval_logits = probe  # type: ignore[method-assign]
  prefix = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
  sampler.generate(
      model, num_samples=1, num_steps=8, eps=1e-3, inject_bos=False,
      prefix_ids=prefix, max_new_tokens=16, greedy=True)
  assert not seen_beyond, seen_beyond


def main() -> None:
  tests = [
      test_uniform_subwindow_active_end_is_window_end,
      test_next_subwindow_forward_excludes_later_unif,
      test_slow_reference_matches_fast_ucc_confidence,
      test_slow_reference_matches_fast_ucc_ltr,
      test_force_max_tie_breaks_leftmost,
      test_thr1_edge_force_max_dominated_and_exact_one,
  ]
  failed = 0
  for fn in tests:
    try:
      fn()
      print(f'PASS  {fn.__name__}')
    except Exception as e:  # noqa: BLE001
      failed += 1
      print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
  if failed:
    raise SystemExit(f'{failed}/{len(tests)} failed')
  print(f'All {len(tests)} slow-reference / sub-window probes passed.')


if __name__ == '__main__':
  main()
