"""Tests for shared decode profiles + samples.meta reuse gate."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from discrete_diffusion.evaluations.decode_profiles import (
    DECODE_PROFILES,
    LM_EVAL_DECODE_PROFILES,
    infer_sample_mode,
    profile_overrides,
    samples_reusable,
    should_reuse_samples,
    write_samples_meta,
)
from discrete_diffusion.evaluations.generative_ppl import _eos_token_stats
import numpy as np


def test_decode_profiles_shared_keys():
  assert set(DECODE_PROFILES) == {
      'baseline', 'hierarchical', 'hubmatch', 'dual_cache'}
  assert set(LM_EVAL_DECODE_PROFILES) == set(DECODE_PROFILES)
  baseline = profile_overrides('baseline')
  assert any('use_arpc=false' in t for t in baseline)
  assert any('unmask_threshold=null' in t for t in baseline)
  assert any('use_block_cache=false' in t for t in baseline)
  hub = profile_overrides('hubmatch')
  assert any('single_stream_decode=true' in t for t in hub)
  assert any('sub_block_size=8' in t for t in hub)
  assert any('use_block_cache=false' in t for t in hub)
  assert any('ban_mask_pad_logits=false' in t for t in hub)
  assert LM_EVAL_DECODE_PROFILES['hubmatch'].get('ban_mask_pad_logits') is False
  assert LM_EVAL_DECODE_PROFILES['dual_cache'].get('ban_mask_pad_logits') is False
  assert profile_overrides('keep') == []


def test_infer_sample_mode_conversion_vs_native():
  conv = OmegaConf.create({
      'data': {'train': 'nemotron_sft', 'tokenizer_name_or_path': 'Qwen/Qwen2.5-1.5B-Instruct'},
      'line': 'ar2block',
  })
  native = OmegaConf.create({
      'data': {'train': 'openwebtext', 'tokenizer_name_or_path': 'gpt2'},
      'line': 'block',
  })
  assert infer_sample_mode(conv) == 'conversion_free'
  assert infer_sample_mode(native) == 'native_free'


def test_samples_reusable_meta_gate(tmp_path: Path):
  samples = tmp_path / 'samples.pt'
  samples.write_bytes(b'fake')
  ckpt = tmp_path / 'model.ckpt'
  ckpt.write_bytes(b'x')
  assert not samples_reusable(
      samples,
      sample_mode='conversion_free',
      decode_profile='baseline',
      checkpoint_path=ckpt,
  )
  write_samples_meta(samples, {
      'checkpoint_path': str(ckpt.resolve()),
      'sample_mode': 'conversion_free',
      'decode_profile': 'baseline',
  })
  assert samples_reusable(
      samples,
      sample_mode='conversion_free',
      decode_profile='baseline',
      checkpoint_path=ckpt,
  )
  assert not samples_reusable(
      samples,
      sample_mode='native_free',
      decode_profile='baseline',
      checkpoint_path=ckpt,
  )
  assert not should_reuse_samples(
      samples,
      checkpoint_path=ckpt,
      sample_mode='conversion_free',
      decode_profile='baseline',
      force_regen=True,
  )


def test_eos_token_stats_honesty():
  eos = 99
  rows = np.array([[1, 2, eos, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int64)
  stats = _eos_token_stats(rows, eos)
  assert stats['eos_rate'] == 0.5
  assert stats['mean_tokens_before_eos'] == 4.0  # (3+5)/2
  assert stats['raw_seq_len'] == 5


def test_trim_still_exported():
  from discrete_diffusion.evaluations.generative_ppl import _trim_token_rows_at_eos
  eos = 99
  rows = np.array([[1, 2, eos, 3]], dtype=np.int64)
  out = _trim_token_rows_at_eos(rows, eos)
  assert out.shape[1] == 3


def test_generate_samples_profile_wins_over_nested_sampling():
  from discrete_diffusion.evaluations.generate_samples import (
      _collect_sampling_overrides,
  )
  cfg = OmegaConf.create({
      'decode_profile': 'baseline',
      'sampling': {
          'use_arpc': True,
          'unmask_threshold': 0.9,
          'use_block_cache': True,
          'hierarchical_kv': True,
      },
      'hydra_overrides': [],
  })
  tokens = _collect_sampling_overrides(cfg)
  joined = ' '.join(tokens)
  assert 'sampling.use_arpc=false' in joined
  assert 'sampling.unmask_threshold=null' in joined
  assert 'sampling.use_block_cache=false' in joined
