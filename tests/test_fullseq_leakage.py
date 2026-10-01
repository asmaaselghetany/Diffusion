"""Full-seq (C3) leakage probes — not copies of the block-UCC tests.

C3 trains/decodes with ``block_size == seq_len`` under Hub single-stream
``eval_block_diff_mask``. That mask is **one block**: every position attends
every other position in the sequence. So "future Unif inside the block is
invisible" is **false by design** — unlike Unif sub8, where
``_truncated_active_end`` clamps to the window.

What must still hold for C3:
  - No reference / GT tokens in the attention input (only prompt + model state)
  - ``block_eval_logits`` truncates at ``active_len`` (API contract)
  - ``generate()`` still has no answer kwargs
  - Committed sticky sites (if UCC is used on C3) still never rewrite

Live harness must re-check corrupt-GT / swapped-prompt on ``2125372``;
these CPU probes only lock the geometry difference.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from discrete_diffusion.models.block_mask import (
    build_eval_block_bool_mask,
    build_block_diff_bool_mask,
)
from discrete_diffusion.sampling.block_sampler import BlockSampler


def test_fullseq_eval_mask_is_fully_dense_within_one_block():
  """block_size == seq_len → every query sees every key (Hub one-block)."""
  n = 16
  m = build_eval_block_bool_mask(n, block_size=n)
  assert m.shape == (n, n)
  assert bool(m.all()), 'full-seq eval mask must be all-True within the block'


def test_block_sub8_eval_mask_hides_later_blocks():
  """Contrast: multi-block eval mask does NOT see later blocks."""
  n, bs = 16, 8
  m = build_eval_block_bool_mask(n, block_size=bs)
  # q in block 0 must not see kv in block 1
  assert not bool(m[0:bs, bs:].any())
  # q in block 1 may see block 0
  assert bool(m[bs:, :bs].all())


def test_fullseq_dual_train_mask_still_blocks_same_index_xt_x0():
  """Train dual-stream geometry (if ever used) must not leak x0[i] to xt[i]."""
  n, b = 16, 16  # one block
  m = build_block_diff_bool_mask(n, b)
  for i in range(n):
    assert not m[i, n + i].item(), f'xt[{i}] must not attend x0[{i}]'


class _FSBackbone(nn.Module):
  def __init__(self, vocab: int, block_size: int):
    super().__init__()
    self.block_size = block_size
    self.vocab = vocab

  def block_eval_logits(self, xt, *, active_len=None, block_size=None, **kw):
    del kw, block_size
    a = int(active_len) if active_len is not None else xt.shape[1]
    # Only :a may affect logits (Hub contract).
    ids = xt[:, :a]
    b, l = xt.shape
    logits = torch.zeros(b, l, self.vocab)
    for i in range(min(a, l)):
      peak = int(ids[0, : i + 1].sum().item()) % (self.vocab - 1)
      logits[:, i, peak] = 5.0
    return logits


class _FSTrainer(nn.Module):
  def __init__(self, n=32, vocab=64):
    super().__init__()
    self.num_tokens = n
    self.block_size = n  # C3 geometry
    self.vocab_size = vocab
    self.mask_id = vocab - 1
    self.sampling_eps = 1e-3
    self.forward_process_name = 'uniform'
    self.backbone = _FSBackbone(vocab, block_size=n)
    self.tokenizer = SimpleNamespace(
        bos_token_id=0, eos_token_id=2, pad_token_id=3)
    from discrete_diffusion.noise_schedules.log_linear import LogLinear
    self.noise = LogLinear(eps=1e-3)

  def backbone_logits(self, xt, x0, active_len=None, **kwargs):
    del x0, kwargs
    return self.backbone.block_eval_logits(xt, active_len=active_len)

  def prior_sample(self, *batch_dims):
    size = batch_dims if len(batch_dims) > 1 else batch_dims[0]
    return torch.randint(0, self.vocab_size - 1, size, dtype=torch.int64)


def _cfg(**kw):
  base = dict(
      steps=4, inject_bos=False, use_float64=False,
      use_arpc=False, arpc_mode='simplified',
      arpc_prefix_frac=0.25, arpc_resample_tau=0.5,
      arpc_corruption_mode='divergence',
      arpc_divergence_measure='kld',
      arpc_diffusion_metric='confidence',
      arpc_ar_metric='nll',
      arpc_warmup_steps=0, arpc_guide_every=1,
      arpc_temperature=1.0, arpc_use_prefix_fill=None,
      hierarchical_kv=True, use_block_cache=False,
      single_stream_decode=True, sub_block_size=None,
      align_shift_logits=None, ar_block_bridge=None,
      ban_mask_pad_logits=True, pad_after_eos=False, stop_on_eos=False,
      greedy=False, p_nucleus=1.0, unmask_threshold=None,
      x0_temperature=1.0, posterior_sampler='fast', track_revisions=False,
      uniform_confidence_sticky=False, sticky_min_conf=0.0,
      uniform_commit_revise=False, uniform_commit_revise_tau=0.25,
      uniform_commit_random=False, uniform_commit_order='confidence',
      allow_full_seq_decode=True,
  )
  base.update(kw)
  return SimpleNamespace(
      algo=SimpleNamespace(forward_process_name='uniform'),
      sampling=SimpleNamespace(**base),
  )


def test_fullseq_logits_ignore_tokens_beyond_active_len():
  """API contract: poisoning xt[active_len:] must not move logits[:active_len]."""
  model = _FSTrainer(n=32)
  sampler = BlockSampler(_cfg())
  xt = torch.randint(0, 30, (1, 32))
  x0 = xt.clone()
  active_end = 16
  base, _ = sampler._logits(
      model, xt, x0, active_end=active_end, window=(0, 16))
  xt2 = xt.clone()
  xt2[:, active_end:] = (xt2[:, active_end:] + 11) % 30
  # Plant fake GT in x0 as well — single-stream must ignore it.
  x0_gt = torch.full_like(x0, 27)
  mut, _ = sampler._logits(
      model, xt2, x0_gt, active_end=active_end, window=(0, 16))
  assert torch.equal(base[:, :active_end], mut[:, :active_end])


def test_fullseq_generate_no_gt_kwargs_and_prefix_only():
  import inspect
  sig = inspect.signature(BlockSampler.generate)
  assert 'answer' not in sig.parameters and 'target' not in sig.parameters
  model = _FSTrainer(n=32)
  sampler = BlockSampler(_cfg(allow_full_seq_decode=True))
  prefix = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
  out = sampler.generate(
      model, num_samples=1, num_steps=4, eps=1e-3, inject_bos=False,
      prefix_ids=prefix, max_new_tokens=8, greedy=False)
  assert torch.equal(out[:, :8], prefix)


def main() -> None:
  tests = [
      test_fullseq_eval_mask_is_fully_dense_within_one_block,
      test_block_sub8_eval_mask_hides_later_blocks,
      test_fullseq_dual_train_mask_still_blocks_same_index_xt_x0,
      test_fullseq_logits_ignore_tokens_beyond_active_len,
      test_fullseq_generate_no_gt_kwargs_and_prefix_only,
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
  print(f'All {len(tests)} full-seq probes passed.')


if __name__ == '__main__':
  main()
