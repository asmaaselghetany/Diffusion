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
      'baseline', 'full_seq_dual', 'hierarchical', 'hierarchical_ancestral',
      'hierarchical_ancestral_t01',
      'hierarchical_ss', 'hierarchical_ss_ancestral', 'hierarchical_arpc',
      'hierarchical_arpc_t01', 'hierarchical_quiet', 'hubmatch',
      'dual_cache', 'uniform_dual', 'uniform_dual_random', 'ucc_l2r_sub8',
      'ss_quiet_ancestral',
      'uniform_commit', 'uniform_commit_t1',
      'uniform_commit_ss'}
  assert set(LM_EVAL_DECODE_PROFILES) == set(DECODE_PROFILES)
  # baseline pins = BlockGen mid-α ancestral skeleton (= hierarchical).
  baseline = profile_overrides('baseline')
  assert any('use_arpc=false' in t for t in baseline)
  assert any('use_block_cache=false' in t for t in baseline)
  assert any('hierarchical_kv=true' in t for t in baseline)
  assert any('unmask_threshold=0.9' in t for t in baseline)
  assert any('single_stream_decode=false' in t for t in baseline)
  assert any('greedy=true' in t for t in baseline)
  hub = profile_overrides('hubmatch')
  assert any('single_stream_decode=true' in t for t in hub)
  assert any('sub_block_size=8' in t for t in hub)
  assert any('use_block_cache=false' in t for t in hub)
  assert any('ban_mask_pad_logits=false' in t for t in hub)
  assert any('unmask_threshold=0.9' in t for t in hub)
  assert LM_EVAL_DECODE_PROFILES['hubmatch'].get('ban_mask_pad_logits') is False
  assert LM_EVAL_DECODE_PROFILES['hubmatch'].get('unmask_threshold') == 0.9
  assert LM_EVAL_DECODE_PROFILES['hubmatch'].get('clear_unmask_threshold') is False
  assert LM_EVAL_DECODE_PROFILES['dual_cache'].get('ban_mask_pad_logits') is False
  assert LM_EVAL_DECODE_PROFILES['dual_cache'].get('unmask_threshold') == 1.0
  assert LM_EVAL_DECODE_PROFILES['dual_cache'].get('clear_unmask_threshold') is False
  dc = profile_overrides('dual_cache')
  assert any('unmask_threshold=1.0' in t for t in dc)
  assert any('use_block_cache=true' in t for t in dc)
  ss = profile_overrides('hierarchical_ss')
  assert any('single_stream_decode=true' in t for t in ss)
  hier = profile_overrides('hierarchical')
  assert any('single_stream_decode=false' in t for t in hier)
  assert any('unmask_threshold=0.9' in t for t in hier)
  assert any('greedy=true' in t for t in hier)
  assert profile_overrides('baseline') == profile_overrides('hierarchical')
  anc = profile_overrides('hierarchical_ancestral')
  assert any('unmask_threshold=null' in t for t in anc)
  assert any('greedy=false' in t for t in anc)
  assert any('x0_temperature=1.0' in t for t in anc)
  assert any('single_stream_decode=false' in t for t in anc)
  anc_t01 = profile_overrides('hierarchical_ancestral_t01')
  assert any('x0_temperature=0.1' in t for t in anc_t01)
  assert any('use_arpc=false' in t for t in anc_t01)
  assert any('single_stream_decode=false' in t for t in anc_t01)
  assert any('sub_block_size=null' in t for t in anc_t01)
  # Tier-2 open-loop 2×2: rows differ only in T; cols only in packing.
  def _pin_map(name):
    d = {}
    for tok in profile_overrides(name):
      k, v = tok.split('=', 1)
      d[k.replace('sampling.', '')] = v
    return d
  a = _pin_map('hierarchical_ancestral')
  a01 = _pin_map('hierarchical_ancestral_t01')
  ss = _pin_map('hierarchical_ss_ancestral')
  ssq = _pin_map('ss_quiet_ancestral')
  for left, right in ((a, a01), (ss, ssq)):
    assert {k: left[k] for k in left if k != 'x0_temperature'} == {
        k: right[k] for k in right if k != 'x0_temperature'}
    assert left['x0_temperature'] == '1.0'
    assert right['x0_temperature'] == '0.1'
  for top, bot in ((a, ss), (a01, ssq)):
    assert top['single_stream_decode'] == 'false'
    assert bot['single_stream_decode'] == 'true'
    assert top['sub_block_size'] == 'null'
    assert bot['sub_block_size'] == '8'
    assert top['x0_temperature'] == bot['x0_temperature']
  arpc = profile_overrides('hierarchical_arpc')
  assert any('use_arpc=true' in t for t in arpc)
  assert any('single_stream_decode=false' in t for t in arpc)
  assert any('x0_temperature=1.0' in t for t in arpc)
  assert LM_EVAL_DECODE_PROFILES['hierarchical_arpc']['use_arpc'] is True
  quiet = profile_overrides('hierarchical_quiet')
  assert any('use_arpc=true' in t for t in quiet)
  assert any('x0_temperature=0.1' in t for t in quiet)
  assert any('arpc_temperature=0.1' in t for t in quiet)
  assert LM_EVAL_DECODE_PROFILES['hierarchical_quiet']['x0_temperature'] == 0.1
  assert LM_EVAL_DECODE_PROFILES['hierarchical_quiet']['arpc_temperature'] == 0.1
  t01 = profile_overrides('hierarchical_arpc_t01')
  assert any('x0_temperature=0.1' in t for t in t01)
  assert any('arpc_temperature=0.1' in t for t in t01)
  assert any('single_stream_decode=false' in t for t in t01)
  assert any('single_stream_decode=true' in t for t in quiet)
  ud = profile_overrides('uniform_dual')
  assert any('uniform_confidence_sticky=true' in t for t in ud)
  assert any('sticky_min_conf=0.0' in t for t in ud)
  assert any('uniform_commit_revise=false' in t for t in ud)
  assert any('unmask_threshold=1.0' in t for t in ud)
  assert any('single_stream_decode=true' in t for t in ud)
  assert any('sub_block_size=8' in t for t in ud)
  assert any('greedy=true' in t for t in ud)
  assert LM_EVAL_DECODE_PROFILES['uniform_dual']['uniform_confidence_sticky'] is True
  udr = profile_overrides('uniform_dual_random')
  assert any('uniform_commit_random=true' in t for t in udr)
  assert any('unmask_threshold=1.0' in t for t in udr)
  assert any('sub_block_size=8' in t for t in udr)
  assert any('x0_temperature=0.1' in t for t in profile_overrides('ss_quiet_ancestral'))
  assert any('use_arpc=false' in t for t in profile_overrides('ss_quiet_ancestral'))
  assert any('uniform_confidence_sticky=false' in t for t in profile_overrides('ss_quiet_ancestral'))
  assert any('sub_block_size=8' in t for t in profile_overrides('ss_quiet_ancestral'))
  assert any('unmask_threshold=null' in t for t in profile_overrides('ss_quiet_ancestral'))
  # DualCache twin ≠ bake-off UC (B1 thr=0.9).
  uc = profile_overrides('uniform_commit')
  assert uc != ud
  assert any('unmask_threshold=0.9' in t for t in uc)
  assert any('single_stream_decode=false' in t for t in uc)
  assert any('sub_block_size=null' in t for t in uc)
  assert any('uniform_commit_revise=false' in t for t in uc)
  assert any('greedy=true' in t for t in uc)
  uc_ss = profile_overrides('uniform_commit_ss')
  assert any('single_stream_decode=true' in t for t in uc_ss)
  assert any('sub_block_size=8' in t for t in uc_ss)
  assert any('unmask_threshold=0.9' in t for t in uc_ss)
  # full_seq_dual must clear sticky UCC + pin posterior like hierarchical.
  fsd = profile_overrides('full_seq_dual')
  assert any('allow_full_seq_decode=true' in t for t in fsd)
  assert any('hierarchical_kv=false' in t for t in fsd)
  assert any('posterior_sampler=fast' in t for t in fsd)
  assert any('uniform_confidence_sticky=false' in t for t in fsd)
  assert any('uniform_commit_revise=false' in t for t in fsd)
  assert profile_overrides('keep') == []


def test_coerce_baseline_blockgen_skeleton():
  from discrete_diffusion.evaluations.decode_profiles import (
      coerce_profile_for_forward,
  )
  # baseline → hierarchical on every arm (UCC is never a silent remap).
  assert coerce_profile_for_forward('baseline', 'masked') == 'hierarchical'
  assert coerce_profile_for_forward('baseline', 'hybrid') == 'hierarchical'
  assert coerce_profile_for_forward('baseline', None) == 'hierarchical'
  assert coerce_profile_for_forward('baseline', 'uniform') == 'hierarchical'
  # Named profiles keep their identity — UCC only when asked for explicitly.
  assert coerce_profile_for_forward('hubmatch', 'uniform') == 'hubmatch'
  assert coerce_profile_for_forward('dual_cache', 'uniform') == 'dual_cache'
  assert coerce_profile_for_forward('dual_cache', 'masked') == 'dual_cache'
  assert coerce_profile_for_forward('hubmatch', 'masked') == 'hubmatch'
  assert coerce_profile_for_forward('hierarchical', 'uniform') == 'hierarchical'
  assert coerce_profile_for_forward('hierarchical_ss', 'uniform') == 'hierarchical_ss'
  assert coerce_profile_for_forward('uniform_dual', 'uniform') == 'uniform_dual'
  assert coerce_profile_for_forward('uniform_commit', 'uniform') == 'uniform_commit'
  assert coerce_profile_for_forward('uniform_commit_t1', 'uniform') == 'uniform_commit_t1'
  assert coerce_profile_for_forward('hierarchical', 'masked') == 'hierarchical'
  # Ablation escape: keep full-seq dual when explicitly allowed.
  assert coerce_profile_for_forward(
      'baseline', 'masked', allow_full_seq=True) == 'full_seq_dual'
  assert coerce_profile_for_forward(
      'full_seq_dual', 'uniform', allow_full_seq=True) == 'full_seq_dual'


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
  # Meta stores the coerced paper-meter name (baseline → hierarchical).
  write_samples_meta(samples, {
      'checkpoint_path': str(ckpt.resolve()),
      'sample_mode': 'conversion_free',
      'decode_profile': 'hierarchical',
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


def test_unigram_shannon_entropy_uniform_and_degenerate():
  from discrete_diffusion.evaluations.generative_ppl import (
      _unigram_shannon_entropy,
  )
  # Two equally likely tokens → ln(2)
  h = _unigram_shannon_entropy([1, 2, 1, 2])
  assert abs(h - np.log(2.0)) < 1e-6
  assert _unigram_shannon_entropy([]) == 0.0
  assert _unigram_shannon_entropy([7, 7, 7]) == 0.0


def test_unigram_entropy_stats_strips_prefix():
  from discrete_diffusion.evaluations.generative_ppl import (
      _unigram_entropy_stats,
  )

  class _Tok:
    name_or_path = 'fake'

    def encode(self, text, add_special_tokens=False):
      del add_special_tokens
      # One id per whitespace-separated token.
      return [hash(t) % 1000 for t in text.split() if t]

  prefix = 'SYS PREFIX '
  texts = [
      prefix + 'a b c d e f g h',
      prefix + 'a a a a a a a a',  # collapsed
  ]
  stats = _unigram_entropy_stats(texts, _Tok(), prefix_text=prefix)
  assert stats['unigram_entropy_num_samples'] == 2
  assert stats['unigram_entropy_mean'] > 0
  # Collapsed sample pulls mean down vs all-diverse.
  diverse_only = _unigram_entropy_stats(
      [prefix + 'a b c d e f g h'], _Tok(), prefix_text=prefix)
  assert stats['unigram_entropy_mean'] < diverse_only['unigram_entropy_mean']


def test_eos_token_stats_honesty():
  eos = 99
  rows = np.array([[1, 2, eos, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int64)
  stats = _eos_token_stats(rows, eos)
  assert stats['eos_rate'] == 0.5
  assert stats['mean_tokens_before_eos'] == 4.0  # (3+5)/2
  assert stats['raw_seq_len'] == 5


def test_eos_token_stats_skips_prefix_eos():
  """Chat conversion_free prefixes embed eos/im_end; ignore until after prefix."""
  eos = 99
  # prefix_len=3 covers [1, eos, 2]; first *generation* eos at index 5
  rows = np.array([[1, eos, 2, 3, 4, eos, 7]], dtype=np.int64)
  stats = _eos_token_stats(rows, eos, prefix_len=3)
  assert stats['eos_rate'] == 1.0
  assert stats['mean_tokens_before_eos'] == 6.0
  assert stats['prefix_len'] == 3


def test_trim_still_exported():
  from discrete_diffusion.evaluations.generative_ppl import _trim_token_rows_at_eos
  eos = 99
  rows = np.array([[1, 2, eos, 3]], dtype=np.int64)
  out = _trim_token_rows_at_eos(rows, eos)
  assert out.shape[1] == 3


def test_trim_skips_prefix_eos():
  from discrete_diffusion.evaluations.generative_ppl import _trim_token_rows_at_eos
  eos = 99
  rows = np.array([[1, eos, 2, 3, eos, 9]], dtype=np.int64)
  out = _trim_token_rows_at_eos(rows, eos, prefix_len=2)
  assert out[0].tolist() == [1, eos, 2, 3, eos]


def test_truncate_im_end_skips_prefix():
  from discrete_diffusion.evaluations.generate_samples import _truncate_im_end
  prefix = '<|im_start|>system\nHi.<|im_end|>\n<|im_start|>assistant\n'
  body = 'Hello world.<|im_end|>\nTRAILING JUNK'
  text = prefix + body
  out = _truncate_im_end(text, prefix_text=prefix)
  assert out == prefix + 'Hello world.<|im_end|>'
  assert 'TRAILING' not in out
  # Without prefix_text, legacy behavior truncates at first im_end (system).
  legacy = _truncate_im_end(text)
  assert legacy == '<|im_start|>system\nHi.<|im_end|>'


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
          'hierarchical_kv': False,
          'single_stream_decode': False,
      },
      'hydra_overrides': [],
  })
  tokens = _collect_sampling_overrides(cfg)
  joined = ' '.join(tokens)
  # Coerce baseline → hierarchical (BlockGen packing + confidence commit).
  assert cfg.decode_profile == 'hierarchical'
  assert 'sampling.use_arpc=false' in joined
  assert 'sampling.unmask_threshold=0.9' in joined
  assert 'sampling.greedy=true' in joined
  assert 'sampling.use_block_cache=false' in joined
  assert 'sampling.hierarchical_kv=true' in joined
  assert 'sampling.single_stream_decode=false' in joined
