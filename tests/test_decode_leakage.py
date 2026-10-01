"""Part-1 decode leakage / correctness probes (CPU, no checkpoint).

Order matches the research checklist:
  1. Corrupt-GT into x0 / side channels must not change window logits
  3. Future-visibility: mutate tokens beyond active_end / window
  5. Clean-stream source: prefix fed to packing is model commits only
  4. Committed-token invariant under UCC (and masked remask twin)

These are homemade-decode legitimacy checks, not goodness claims.
Part-2 (matched-NFE baselines, selection, replication) lives in the bake.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from discrete_diffusion.sampling.block_sampler import BlockSampler


class _ProbeBackbone(nn.Module):
  """Logits that depend only on the tokens the real APIs are allowed to see."""

  def __init__(self, vocab: int, block_size: int = 8):
    super().__init__()
    self.block_size = block_size
    self.vocab = vocab
    self.calls: list[dict] = []

  def block_eval_logits(self, xt, *, active_len=None, block_size=None, **kw):
    del kw, block_size
    a = int(active_len) if active_len is not None else xt.shape[1]
    seen = xt[:, :a].clone()
    self.calls.append({'api': 'block_eval_logits', 'seen': seen, 'a': a})
    return self._ids_to_logits(seen, out_len=xt.shape[1])

  def block_gen_logits(
      self, prefix_ids, xt_block, *, block_size=None, x0_causal=False, **kw):
    del kw, block_size, x0_causal
    self.calls.append({
        'api': 'block_gen_logits',
        'prefix': prefix_ids.clone(),
        'xt_block': xt_block.clone(),
    })
    return self._ids_to_logits(xt_block, out_len=xt_block.shape[1])

  def _ids_to_logits(self, ids: torch.Tensor, *, out_len: int) -> torch.Tensor:
    # Deterministic content-dependent logits so mutations are detectable.
    b, l = ids.shape
    logits = torch.zeros(b, out_len, self.vocab, dtype=torch.float32)
    use = min(l, out_len)
    for i in range(use):
      # Peak at token id itself, nudged by a rolling mix of visible context.
      mix = (ids[:, : i + 1].sum(dim=-1) % (self.vocab - 1)) + 1
      logits[:, i, :] = -10.0
      logits[torch.arange(b), i, mix] = 5.0
      logits[torch.arange(b), i, ids[:, i] % self.vocab] = 8.0
    return logits


class _ProbeTrainer(nn.Module):
  def __init__(self, n=32, vocab=64, mode='uniform', block_size=8):
    super().__init__()
    self.num_tokens = n
    self.block_size = block_size
    self.vocab_size = vocab
    self.mask_id = vocab - 1
    self.sampling_eps = 1e-3
    self.forward_process_name = mode
    self.backbone = _ProbeBackbone(vocab, block_size=block_size)
    self.tokenizer = SimpleNamespace(
        bos_token_id=0, eos_token_id=2, pad_token_id=3)
    from discrete_diffusion.noise_schedules.log_linear import LogLinear
    self.noise = LogLinear(eps=1e-3)

  def backbone_logits(self, xt, x0, active_len=None, **kwargs):
    del kwargs
    a = int(active_len) if active_len is not None else xt.shape[1]
    # Equal-length dual: record both halves (leak surface if x0 holds GT).
    self.backbone.calls.append({
        'api': 'backbone_logits',
        'xt': xt[:, :a].clone(),
        'x0': x0[:, :a].clone(),
        'a': a,
    })
    return self.backbone._ids_to_logits(xt[:, :a], out_len=xt.shape[1])

  def prior_sample(self, *batch_dims):
    size = batch_dims if len(batch_dims) > 1 else batch_dims[0]
    if self.forward_process_name == 'uniform':
      return torch.randint(0, self.vocab_size - 1, size, dtype=torch.int64)
    return torch.full(size, self.mask_id, dtype=torch.int64)


def _sampling(**kw):
  base = dict(
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
      hierarchical_kv=True, use_block_cache=False,
      single_stream_decode=True,
      sub_block_size=8,
      align_shift_logits=None,
      ar_block_bridge=None,
      ban_mask_pad_logits=True,
      pad_after_eos=False,
      stop_on_eos=False,
      greedy=True,
      p_nucleus=1.0,
      unmask_threshold=1.0,
      x0_temperature=1.0,
      posterior_sampler='fast',
      track_revisions=False,
      uniform_confidence_sticky=True,
      sticky_min_conf=0.0,
      uniform_commit_revise=False,
      uniform_commit_revise_tau=0.25,
      uniform_commit_random=False,
      uniform_commit_order='confidence',
      allow_full_seq_decode=False,
  )
  base.update(kw)
  return SimpleNamespace(**base)


def _cfg(mode: str, **sampling_kw):
  return SimpleNamespace(
      algo=SimpleNamespace(forward_process_name=mode),
      sampling=_sampling(**sampling_kw),
  )


# ---------------------------------------------------------------------------
# 5. Clean-stream source (most likely subtle leak)
# ---------------------------------------------------------------------------

def test_ucc_single_stream_never_feeds_x0_to_backbone():
  """ucc_thr1_sub8 path: block_eval_logits(xt) only — no clean N2C half."""
  model = _ProbeTrainer(mode='uniform')
  sampler = BlockSampler(_cfg(
      'uniform',
      single_stream_decode=True,
      use_block_cache=False,
      hierarchical_kv=True,
      uniform_confidence_sticky=True,
      unmask_threshold=1.0,
      sub_block_size=8,
  ))
  prefix = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
  model.backbone.calls.clear()
  out = sampler.generate(
      model, num_samples=1, num_steps=8, eps=1e-3, inject_bos=False,
      prefix_ids=prefix, max_new_tokens=16, greedy=True)
  assert out.shape[1] == model.num_tokens
  apis = {c['api'] for c in model.backbone.calls}
  assert 'block_eval_logits' in apis
  assert 'backbone_logits' not in apis, (
      'UCC single-stream decode fell back to dual backbone_logits — '
      'clean-stream surface reopened')
  assert 'block_gen_logits' not in apis


def test_hierarchical_block_gen_prefix_is_committed_xt():
  """Clean half of block_gen packing must equal xt/x0 prefix (model commits)."""
  model = _ProbeTrainer(mode='uniform')

  # Force hierarchical block_gen path: no single_stream / DualCache eval API.
  class _GenOnly(_ProbeBackbone):
    def block_eval_logits(self, *args, **kwargs):  # noqa: ANN002
      raise AssertionError('block_eval_logits must not run on this path')

  model.backbone = _GenOnly(model.vocab_size, block_size=model.block_size)
  sampler = BlockSampler(_cfg(
      'uniform',
      single_stream_decode=False,
      use_block_cache=False,
      hierarchical_kv=True,
      uniform_confidence_sticky=False,
      unmask_threshold=None,
      sub_block_size=None,
      greedy=False,
  ))
  n, bs = model.num_tokens, model.block_size
  start, end = bs, 2 * bs
  prefix = torch.arange(bs).unsqueeze(0)
  future = torch.full((1, n - end), 3)
  current = torch.randint(0, model.vocab_size - 1, (1, bs))
  xt = torch.cat([prefix, current, future], dim=-1)
  x0 = xt.clone()
  # Plant a fake "GT answer" in x0 future — must never appear in logged prefix.
  gt = torch.full((1, n - end), 42)
  x0[:, end:] = gt

  model.backbone.calls.clear()
  sampler._denoise_block(
      model, xt.clone(), x0.clone(), start, end, num_steps=2, eps=1e-3)
  gen_calls = [c for c in model.backbone.calls if c['api'] == 'block_gen_logits']
  assert gen_calls, 'expected block_gen_logits packing'
  for c in gen_calls:
    assert torch.equal(c['prefix'], prefix), (
        'clean prefix drifted from committed tokens — possible GT/N2C leak')
    assert not (c['prefix'] == 42).any()


# ---------------------------------------------------------------------------
# 1. Corrupt-GT test (logits identity)
# ---------------------------------------------------------------------------

def test_corrupt_gt_in_x0_does_not_change_single_stream_logits():
  """Plant answer tokens in x0; single-stream window logits must be identical."""
  model = _ProbeTrainer(mode='uniform')
  sampler = BlockSampler(_cfg(
      'uniform',
      single_stream_decode=True,
      hierarchical_kv=True,
      use_block_cache=False,
      uniform_confidence_sticky=True,
      unmask_threshold=1.0,
  ))
  n = model.num_tokens
  xt = torch.randint(0, model.vocab_size - 1, (1, n))
  x0 = xt.clone()
  window = (8, 16)
  active_end = 16

  logits_clean, _ = sampler._logits(
      model, xt, x0, active_end=active_end, window=window)

  x0_gt = x0.clone()
  # Classic N2C leak surface: put "answer" on the clean half everywhere.
  x0_gt[:, :] = (x0_gt + 17) % (model.vocab_size - 1)
  logits_gt, _ = sampler._logits(
      model, xt, x0_gt, active_end=active_end, window=window)

  assert torch.equal(logits_clean, logits_gt), (
      'single-stream logits moved after corrupting x0 — decode reads clean half')


def test_generate_has_no_answer_kwarg():
  """lm-eval path: generate only conditions on prefix_ids (prompt)."""
  import inspect
  sig = inspect.signature(BlockSampler.generate)
  forbidden = {
      'answer', 'answers', 'target', 'targets', 'reference', 'references',
      'label', 'labels', 'gt', 'ground_truth', 'x0_gt', 'clean_ids',
  }
  overlap = forbidden & set(sig.parameters)
  assert not overlap, f'generate accepts GT kwargs: {overlap}'


# ---------------------------------------------------------------------------
# 3. Future-visibility
# ---------------------------------------------------------------------------

def test_future_tokens_do_not_affect_single_stream_logits():
  model = _ProbeTrainer(mode='uniform')
  sampler = BlockSampler(_cfg(
      'uniform',
      single_stream_decode=True,
      hierarchical_kv=True,
      use_block_cache=False,
  ))
  n = model.num_tokens
  xt = torch.randint(0, model.vocab_size - 1, (1, n))
  x0 = xt.clone()
  window = (8, 16)
  active_end = 16

  base, _ = sampler._logits(
      model, xt, x0, active_end=active_end, window=window)

  xt_f = xt.clone()
  xt_f[:, active_end:] = (xt_f[:, active_end:] + 23) % (model.vocab_size - 1)
  x0_f = x0.clone()
  x0_f[:, active_end:] = (x0_f[:, active_end:] + 29) % (model.vocab_size - 1)
  # Also scribble not-yet-committed sites beyond the current sub-window
  # but inside active_end (attention-block span may be larger than sub8).
  xt_f[:, 16:24] = (xt_f[:, 16:24] + 5) % (model.vocab_size - 1)

  mut, _ = sampler._logits(
      model, xt_f, x0_f, active_end=active_end, window=window)
  # Positions inside the visible span must match; beyond may be padded zeros.
  assert torch.equal(base[:, :active_end], mut[:, :active_end]), (
      'logits inside active_end changed after mutating future tokens')


def test_denoise_restore_keeps_future_and_prefix():
  """Existing restore contract (also in test_block_sampler) — keep for Part-1."""
  model = _ProbeTrainer(mode='uniform', n=16)
  sampler = BlockSampler(_cfg(
      'uniform',
      uniform_confidence_sticky=False,
      unmask_threshold=None,
      single_stream_decode=True,
      sub_block_size=None,
  ))
  n, bs = 16, 8
  start, end = bs, 2 * bs
  prefix = torch.arange(bs).unsqueeze(0)
  future = torch.full((1, n - end), 3)
  current = torch.full((1, bs), 5)
  xt = torch.cat([prefix, current, future], dim=-1)
  x0 = xt.clone()

  def scribble(_m, xt_in, _x0, _t, _dt, **kw):
    del _m, _x0, _t, _dt, kw
    return torch.full_like(xt_in, 7)

  sampler._uniform_step = scribble
  out, x0_out = sampler._denoise_block(
      model, xt.clone(), x0.clone(), start, end, num_steps=2, eps=1e-3)
  assert torch.equal(out[:, :start], prefix)
  assert torch.equal(out[:, end:], future)
  assert torch.equal(x0_out[:, end:], future)


# ---------------------------------------------------------------------------
# 4. Committed-token invariant (UCC + masked remask)
# ---------------------------------------------------------------------------

def test_ucc_committed_invariant_via_step_hook():
  model = _ProbeTrainer(mode='uniform', n=24)
  sampler = BlockSampler(_cfg(
      'uniform',
      single_stream_decode=True,
      hierarchical_kv=True,
      use_block_cache=False,
      uniform_confidence_sticky=True,
      unmask_threshold=1.0,
      sticky_min_conf=0.0,
      sub_block_size=8,
      stop_on_eos=False,
  ))
  violations: list[str] = []
  prev_frozen: dict[tuple[int, int], torch.Tensor] = {}
  prev_unfrozen: dict[tuple[int, int], int] = {}

  def hook(event):
    start = int(event['window_start'])
    end = int(event['window_end'])
    key = (start, end)
    toks = torch.tensor(event['token_ids'], dtype=torch.long)
    frozen_idx = set(event['committed_positions'])
    # Positions committed in prior steps inside this window must be stable.
    if key in prev_frozen:
      old = prev_frozen[key]
      for i in range(start, end):
        if i in frozen_idx and i < old.numel():
          # Once a site appears in newly_committed, later events must keep it.
          pass
      newly = set(event.get('newly_committed_positions') or [])
      # Tokens at previously newly-committed sites must not change.
      for i in getattr(hook, '_ever_committed', set()):
        if start <= i < end and i < toks.numel() and i < old.numel():
          if int(toks[i]) != int(old[i]):
            violations.append(
                f'committed pos {i} changed {int(old[i])}→{int(toks[i])}')
      hook._ever_committed |= newly  # type: ignore[attr-defined]
    else:
      hook._ever_committed = set(event.get('newly_committed_positions') or [])
    # Undecided count in window must not increase.
    frozen = sampler._block_sticky_frozen
    if frozen is not None:
      unf = int((~frozen[0, start:end]).sum().item())
      if key in prev_unfrozen and unf > prev_unfrozen[key]:
        violations.append(
            f'undecided rose in [{start},{end}): '
            f'{prev_unfrozen[key]}→{unf}')
      prev_unfrozen[key] = unf
    prev_frozen[key] = toks.clone()

  hook._ever_committed = set()  # type: ignore[attr-defined]
  sampler.step_hook = hook
  prefix = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
  out = sampler.generate(
      model, num_samples=1, num_steps=8, eps=1e-3, inject_bos=False,
      prefix_ids=prefix, max_new_tokens=16, greedy=True)
  assert not violations, violations
  # Prompt frozen.
  assert torch.equal(out[:, :8], prefix)


def test_masked_confidence_committed_invariant():
  model = _ProbeTrainer(mode='masked', n=24)
  sampler = BlockSampler(_cfg(
      'masked',
      single_stream_decode=True,
      hierarchical_kv=True,
      use_block_cache=False,
      uniform_confidence_sticky=False,
      unmask_threshold=0.9,
      sub_block_size=8,
      stop_on_eos=False,
      greedy=True,
  ))
  ever: set[int] = set()
  last: dict[int, int] = {}
  violations: list[str] = []

  def hook(event):
    toks = event['token_ids']
    for i in event.get('newly_committed_positions') or []:
      ever.add(int(i))
    for i in ever:
      if i >= len(toks):
        continue
      v = int(toks[i])
      if i in last and last[i] != v and v != model.mask_id:
        # Allow first unmask; forbid later rewrite away from committed id.
        if last[i] != model.mask_id:
          violations.append(f'masked commit {i} rewrote {last[i]}→{v}')
      if v != model.mask_id:
        last[i] = v

  sampler.step_hook = hook
  prefix = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
  out = sampler.generate(
      model, num_samples=1, num_steps=8, eps=1e-3, inject_bos=False,
      prefix_ids=prefix, max_new_tokens=16, greedy=True)
  assert not violations, violations
  assert torch.equal(out[:, :8], prefix)
  # No MASK left in the generated span we asked for.
  assert (out[:, 8:24] != model.mask_id).all()


def main() -> None:
  tests = [
      test_generate_has_no_answer_kwarg,
      test_corrupt_gt_in_x0_does_not_change_single_stream_logits,
      test_future_tokens_do_not_affect_single_stream_logits,
      test_denoise_restore_keeps_future_and_prefix,
      test_ucc_single_stream_never_feeds_x0_to_backbone,
      test_hierarchical_block_gen_prefix_is_committed_xt,
      test_ucc_committed_invariant_via_step_hook,
      test_masked_confidence_committed_invariant,
  ]
  failed = 0
  for fn in tests:
    try:
      fn()
      print(f'PASS  {fn.__name__}')
    except Exception as e:  # noqa: BLE001 — report all, don't abort suite
      failed += 1
      print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
  if failed:
    raise SystemExit(f'{failed}/{len(tests)} failed')
  print(f'All {len(tests)} Part-1 probes passed.')


if __name__ == '__main__':
  main()
