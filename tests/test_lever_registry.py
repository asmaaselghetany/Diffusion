"""Lever registry resolve/validate — wrong arm / conflicts must fail loud."""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    'resolve_lever', _REPO / 'tools' / 'resolve_lever.py')
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules['resolve_lever'] = _MOD
_SPEC.loader.exec_module(_MOD)
load_registry = _MOD.load_registry
resolve = _MOD.resolve


@pytest.fixture(scope='module')
def reg():
  return load_registry()


def test_fdllm_preset_masked_ok(reg):
  r = resolve(preset='fdllm', arm='masked', line='ar2block', registry=reg)
  assert 'algo.shift_loss_targets=true' in r['overrides']
  assert 'algo.complementary_masks=true' in r['overrides']
  assert 'algo.mask_schedule=fast_dllm' in r['overrides']


def test_paper_C2_fdllm_is_strict_axis_d(reg):
  """Tab-2 C2_fdllm = shift+comp only (no mask_schedule confound)."""
  r = resolve(preset='C2_fdllm', arm='masked', line='ar2block', registry=reg)
  assert 'algo.shift_loss_targets=true' in r['overrides']
  assert 'algo.complementary_masks=true' in r['overrides']
  assert 'algo.complementary_batching=fused' in r['overrides']
  assert 'algo.mask_schedule=fast_dllm' not in r['overrides']


def test_paper_C2_fdllm_full_adds_schedule(reg):
  r = resolve(
      preset='C2_fdllm_full', arm='masked', line='ar2block', registry=reg)
  assert 'algo.mask_schedule=fast_dllm' in r['overrides']
  assert 'algo.loss_weighting=plain_ce' in r['overrides']
  assert 'algo.hub_struct_attn_only=true' in r['overrides']
  assert 'algo.ignore_bos=false' in r['overrides']


def test_fdllm_confidence_decode_matches_eval_script_thr1(reg):
  r = resolve(
      lever_ids=['fdllm_confidence_decode'], arm='masked', line='ar2block',
      registry=reg)
  assert 'sampling.unmask_threshold=1.0' in r['overrides']
  assert 'sampling.ban_mask_pad_logits=false' in r['overrides']


def test_paper_C0_empty(reg):
  r = resolve(preset='C0', arm='masked', line='ar2block', registry=reg)
  assert r['overrides'] == []


def test_fdllm_preset_uniform_refused(reg):
  with pytest.raises(ValueError, match='only allows arms'):
    resolve(preset='fdllm', arm='uniform', line='ar2block', registry=reg)


def test_arpc_masked_refused(reg):
  with pytest.raises(ValueError, match='only allowed on arms'):
    resolve(
        lever_ids=['mixture_1_32', 'arpc'],
        arm='masked', line='block', registry=reg)


def test_mixture_vs_weights_conflict(reg):
  with pytest.raises(ValueError, match='conflict'):
    resolve(
        lever_ids=['mixture_16_32', 'bg_weights_1_16'],
        arm='uniform', line='block', registry=reg)


def test_arpc_requires_size_1(reg):
  with pytest.raises(ValueError, match='requires'):
    resolve(lever_ids=['arpc'], arm='uniform', line='block', registry=reg)


def test_blockgen_uniform_preset(reg):
  r = resolve(
      preset='blockgen_uniform', arm='uniform', line='block', registry=reg)
  joined = ' '.join(r['overrides'])
  assert 'block_weights=[0.05,0.0,0.0,0.0,0.95]' in joined
  assert 'use_arpc=true' in joined
  assert 'arpc_mode=blockgen' in joined
  assert 'pure_noise_block_sizes' in joined


def test_blockgen_uniform_refuses_ar2block(reg):
  with pytest.raises(ValueError, match='only allows line'):
    resolve(
        preset='blockgen_uniform', arm='uniform', line='ar2block',
        registry=reg)


def test_native_levers_refuse_raw_ar2block(reg):
  with pytest.raises(ValueError, match='family=native'):
    resolve(
        lever_ids=['bg_weights_1_16', 'u_stratified'],
        arm='uniform', line='ar2block', registry=reg)


def test_xfer_preset_allows_native_on_ar2block(reg):
  r = resolve(
      preset='xfer_u_stratified', arm='uniform', line='ar2block',
      registry=reg)
  assert 'u-stratified' in ' '.join(r['overrides'])
  assert any('not a BlockGen paper claim' in w for w in r['warnings'])


def test_b3_arpc_is_scratch_block(reg):
  r = resolve(preset='B3_arpc', arm='uniform', line='block', registry=reg)
  assert 'arpc_mode=blockgen' in ' '.join(r['overrides'])
  with pytest.raises(ValueError, match='only allows line'):
    resolve(preset='B3_arpc', arm='uniform', line='ar2block', registry=reg)


def test_fdllm_levers_refuse_scratch_block(reg):
  with pytest.raises(ValueError, match='family=conversion|only allowed on line'):
    resolve(lever_ids=['shift'], arm='masked', line='block', registry=reg)


def test_c5_joint_ar_preset(reg):
  r = resolve(preset='C5_joint_ar', arm='masked', line='ar2block', registry=reg)
  assert 'algo.joint_ar_alpha=0.3' in r['overrides']


def test_c5_causal_clean_requires_joint(reg):
  r = resolve(
      preset='C5_causal_clean', arm='masked', line='ar2block', registry=reg)
  assert 'algo.joint_ar_alpha=0.3' in r['overrides']
  assert 'algo.causal_clean_stream=true' in r['overrides']


def test_b4_hybrid_preset(reg):
  r = resolve(
      preset='B4_hybrid_p10', arm='hybrid', line='ar2block', registry=reg)
  assert 'algo.hybrid_p_uniform=0.1' in r['overrides']
  assert any('default p_uniform' in w for w in r['warnings'])


def test_b4_hybrid_masked_arm_refused(reg):
  with pytest.raises(ValueError, match='only allows arms'):
    resolve(
        preset='B4_hybrid_p10', arm='masked', line='ar2block', registry=reg)


def test_hierarchical_kv_does_not_force_dual_cache(reg):
  r = resolve(
      lever_ids=['hierarchical_kv'], arm='masked', line='ar2block',
      registry=reg)
  joined = ' '.join(r['overrides'])
  assert 'sampling.hierarchical_kv=true' in joined
  assert 'use_block_cache' not in joined


def test_dual_cache_requires_hierarchical(reg):
  with pytest.raises(ValueError, match='requires'):
    resolve(
        lever_ids=['dual_cache'], arm='masked', line='ar2block', registry=reg)


def test_decode_dual_cache_preset(reg):
  r = resolve(
      preset='decode_dual_cache', arm='masked', line='ar2block', registry=reg)
  joined = ' '.join(r['overrides'])
  assert 'hierarchical_kv=true' in joined
  assert 'use_block_cache=true' in joined


def test_extra_overrides_orphan_root_refused():
  with pytest.raises(ValueError, match='algo. prefix'):
    _MOD.validate_extra_overrides(['shift_loss_targets=true'])
  with pytest.raises(ValueError, match='algo. prefix'):
    _MOD.validate_extra_overrides(['hybrid_decode=masked'])
  with pytest.raises(ValueError, match='algo. prefix'):
    _MOD.validate_extra_overrides(['causal_clean_stream=true'])
  with pytest.raises(ValueError, match='sampling. prefix'):
    _MOD.validate_extra_overrides(['hierarchical_kv=true'])
  with pytest.raises(ValueError, match='sampling. prefix'):
    _MOD.validate_extra_overrides(['use_block_cache=true'])


def test_extra_overrides_unknown_prefix_warns():
  w = _MOD.validate_extra_overrides(['foo.bar=1'])
  assert w and 'unknown prefix' in w[0]


def test_decode_sub_block_arms_match_lever(reg):
  preset = reg['presets']['decode_sub_block']
  lever = reg['levers']['sub_block_8']
  assert preset['arms'] == lever['arms'] == ['masked', 'uniform']


def test_every_lever_appears_in_some_preset(reg):
  used = set()
  for p in (reg.get('presets') or {}).values():
    used |= set(p.get('levers') or [])
  orphans = sorted(set(reg['levers']) - used)
  assert orphans == [], f'orphan levers (add a preset/cell): {orphans}'
