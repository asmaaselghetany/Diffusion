"""Tests for Unifusion-style intra-block attention anneal + revision stats."""

from __future__ import annotations

import torch


def test_intra_block_open_zero_is_causal_within_block():
  from discrete_diffusion.models.block_mask import build_block_diff_bool_mask

  n, bs = 8, 4
  # Full bi
  full = build_block_diff_bool_mask(n, bs, 'cpu', intra_block_open=1.0)
  # Causal within block
  causal = build_block_diff_bool_mask(n, bs, 'cpu', intra_block_open=0.0)
  # Within first xt block positions 0..3: q=2,kv=3 should be False when open=0
  # Indices on 2n grid: xt stream uses 0..n-1
  assert bool(full[2, 3])  # bi allows look-ahead
  assert not bool(causal[2, 3])  # causal forbids
  assert bool(causal[3, 2])  # past ok
  # Across-block offset (xt attends previous x0) still present when open=0
  # q=4 (xt block1), kv=n+0 (x0 block0) — offset_block_causal
  assert bool(causal[4, n + 0])


def test_revision_stats_api_defaults_off():
  from discrete_diffusion.sampling.block_sampler import BlockSampler
  import omegaconf

  cfg = omegaconf.OmegaConf.create({
      'sampling': {
          'steps': 4,
          'inject_bos': True,
          'track_revisions': False,
          'use_arpc': False,
          'stop_on_eos': False,
          'pad_after_eos': False,
      },
      'algo': {'forward_process_name': 'uniform'},
  })
  # Minimal construct — BlockSampler may need more; use __new__
  s = BlockSampler.__new__(BlockSampler)
  s.track_revisions = False
  s._revision_stats = None
  assert s.pop_revision_stats() is None
  s.track_revisions = True
  s.reset_revision_stats()
  s._revision_stats['token_changes'] = 10
  s._revision_stats['token_slots'] = 100
  s._revision_stats['steps_tracked'] = 5
  out = s.pop_revision_stats()
  assert abs(out['revision_rate'] - 0.1) < 1e-9
