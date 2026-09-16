"""Unit tests for align_shift_logits full vs DualCache-window modes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from discrete_diffusion.sampling.block_sampler import BlockSampler
from discrete_diffusion.sampling.shift_logits import align_shift_logits


def test_align_shift_full_mode_dense_uses_left_neighbor():
  """Dense/prefill: aligned[w0] == dense_logits[w0-1] when w0 > 0."""
  torch.manual_seed(0)
  logits = torch.randn(2, 8, 5)
  w0, w1 = 3, 6
  aligned = align_shift_logits(logits, enabled=True, mode='full')
  assert torch.equal(aligned[:, w0], logits[:, w0 - 1])
  assert torch.equal(aligned[:, :1], logits[:, :1])
  assert torch.equal(aligned[:, 1:], logits[:, :-1])
  # mode=None defaults to full; window must not force window-local.
  aligned_default = align_shift_logits(
      logits, enabled=True, window=(w0, w1), mode=None)
  assert torch.equal(aligned_default[:, w0], logits[:, w0 - 1])


def test_align_shift_window_mode_replace_keeps_first_col():
  """DualCache replace / zero-pad: aligned[w0] == win[0], not left pad zero."""
  torch.manual_seed(1)
  b, t, v = 2, 8, 5
  w0, w1 = 3, 6
  win = torch.randn(b, w1 - w0, v)
  logits = torch.zeros(b, t, v)
  logits[:, w0:w1, :] = win

  aligned = align_shift_logits(
      logits, enabled=True, mode='window', window=(w0, w1))
  assert torch.equal(aligned[:, w0], win[:, 0])
  assert not torch.allclose(aligned[:, w0], torch.zeros_like(aligned[:, w0]))
  # Positions after w0 inside the window shift from the prior window col.
  assert torch.equal(aligned[:, w0 + 1:w1], win[:, :-1])
  # Outside the window stays zero (untouched).
  assert torch.equal(aligned[:, :w0], torch.zeros(b, w0, v))
  assert torch.equal(aligned[:, w1:], torch.zeros(b, t - w1, v))


def test_align_shift_full_on_zero_pad_is_the_token0_bug():
  """Document why replace must not use full shift: aligned[w0] becomes zeros."""
  w0, w1 = 3, 6
  win = torch.randn(1, w1 - w0, 4)
  logits = torch.zeros(1, 8, 4)
  logits[:, w0:w1, :] = win
  buggy = align_shift_logits(logits, enabled=True, mode='full')
  assert torch.allclose(buggy[:, w0], torch.zeros(1, 4))


def test_align_shift_disabled_is_identity():
  logits = torch.randn(1, 4, 3)
  out = align_shift_logits(
      logits, enabled=False, mode='window', window=(1, 3))
  assert out is logits or torch.equal(out, logits)


def test_align_shift_window_mode_requires_window():
  with pytest.raises(ValueError, match='requires window'):
    align_shift_logits(torch.randn(1, 4, 2), enabled=True, mode='window')


def test_attention_block_end_ceils_sub_window():
  assert BlockSampler._attention_block_end(24, 32, 128) == 32
  assert BlockSampler._attention_block_end(48, 32, 128) == 64
  assert BlockSampler._attention_block_end(32, 32, 128) == 32
  assert BlockSampler._attention_block_end(130, 32, 128) == 128


def test_truncated_active_end_rounds_when_hierarchical():
  cfg = SimpleNamespace(
      algo=SimpleNamespace(forward_process_name='masked'),
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
          hierarchical_kv=True, use_block_cache=False,
          single_stream_decode=False,
          sub_block_size=8,
          align_shift_logits=True,
          pad_after_eos=True,
          stop_on_eos=True,
          greedy=False,
          p_nucleus=1.0,
          unmask_threshold=None,
      ),
  )
  sampler = BlockSampler(cfg)
  model = SimpleNamespace(block_size=32)
  assert sampler._truncated_active_end(model, end=40, seq_len=128) == 64

  cfg.sampling.hierarchical_kv = False
  sampler2 = BlockSampler(cfg)
  assert sampler2._truncated_active_end(model, end=40, seq_len=128) == 40

  # Confidence / hubmatch scope also ceils without hierarchical_kv alone.
  cfg.sampling.unmask_threshold = 0.9
  sampler3 = BlockSampler(cfg)
  assert sampler3._truncated_active_end(model, end=40, seq_len=128) == 64


def test_logits_reports_full_then_window_for_dual_cache():
  """Prefill → full shift; subsequent replace → window shift.

  Hub replace path only runs when the first small-block token is *not* MASK
  (generation_functions.py ~102); otherwise DualCache is refreshed/prefilled.
  """
  cfg = SimpleNamespace(
      algo=SimpleNamespace(forward_process_name='masked'),
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
          hierarchical_kv=True, use_block_cache=True,
          single_stream_decode=True,
          sub_block_size=None,
          align_shift_logits=True,
          pad_after_eos=True,
          stop_on_eos=True,
          greedy=False,
          p_nucleus=1.0,
          unmask_threshold=None,
      ),
  )
  sampler = BlockSampler(cfg)
  b, a, v = 1, 16, 7
  w0, w1 = 8, 12
  mask_id = 6
  dense = torch.randn(b, a, v)
  replace = torch.zeros(b, a, v)
  replace[:, w0:w1] = torch.randn(b, w1 - w0, v)

  class _BB:
    block_size = 16

    def block_eval_prefill(self, xt, *, active_len, block_size):
      del xt, block_size
      return dense[:, :active_len].clone(), SimpleNamespace(active_len=active_len)

    def block_eval_replace(self, xt, *, active_len, window, cache, block_size):
      del xt, active_len, window, cache, block_size
      return replace.clone()

  model = SimpleNamespace(backbone=_BB(), block_size=16, mask_id=mask_id)
  xt = torch.randint(0, mask_id, (b, 32))  # avoid MASK at window start
  x0 = xt.clone()
  window = (w0, w1)

  logits0, mode0 = sampler._logits(
      model, xt, x0, active_end=a, window=window)
  assert mode0 == 'full'
  assert torch.equal(logits0, dense)
  assert sampler._dual_cache is not None
  assert sampler._dual_cache.active_len == a

  logits1, mode1 = sampler._logits(
      model, xt, x0, active_end=a, window=window)
  assert mode1 == 'window'
  assert torch.equal(logits1, replace)

  prepared = sampler._prepare_masked_logits(
      model, logits1, window=window, shift_mode=mode1)
  # Window-local: first denoise col keeps replace[w0] (MASK ban may rewrite
  # the mask-id channel only).
  assert torch.allclose(
      prepared[:, w0, :mask_id], replace[:, w0, :mask_id])
  assert float(prepared[0, w0, mask_id]) <= -1e5

  # While first token of the window is MASK, Hub refreshes → full/prefill again.
  xt[:, w0] = mask_id
  logits2, mode2 = sampler._logits(
      model, xt, x0, active_end=a, window=window)
  assert mode2 == 'full'
