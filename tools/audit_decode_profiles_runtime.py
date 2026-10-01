#!/usr/bin/env python3
"""Runtime audit: DECODE_PROFILES pins → BlockSampler path (masked/uniform).

# region agent log
Writes NDJSON to the debug session log for hypothesis grading.
# endregion
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

LOG = Path('/e/project1/scifi/elsayed3/.cursor/debug-bec135.log')


def log(hid: str, msg: str, data: dict, run_id: str = 'profile-audit') -> None:
  # region agent log
  payload = {
      'sessionId': 'bec135',
      'runId': run_id,
      'hypothesisId': hid,
      'location': 'tools/audit_decode_profiles_runtime.py',
      'message': msg,
      'data': data,
      'timestamp': int(time.time() * 1000),
  }
  LOG.parent.mkdir(parents=True, exist_ok=True)
  with LOG.open('a', encoding='utf-8') as f:
    f.write(json.dumps(payload, default=str) + '\n')
  # endregion


def main() -> int:
  from omegaconf import OmegaConf

  from discrete_diffusion.evaluations.decode_profiles import (
      DECODE_PROFILES,
      LM_EVAL_DECODE_PROFILES,
      coerce_profile_for_forward,
      profile_overrides,
      _parse_hydra_override_value,
  )
  from discrete_diffusion.sampling.block_sampler import BlockSampler

  def overrides_to_sampling(overrides):
    d = {}
    for tok in overrides:
      k, v = tok.split('=', 1)
      assert k.startswith('sampling.')
      d[k[len('sampling.'):]] = _parse_hydra_override_value(v)
    return d

  def classify(sampler):
    is_masked = sampler.is_masked
    thr = sampler.unmask_threshold
    sticky = sampler.uniform_confidence_sticky
    arpc = sampler.use_arpc and sampler.arpc_mode == 'blockgen'
    # Mirror generate(): thr without sticky on Unif is ignored → fixed-N ancestral.
    dead_thr = (not is_masked and thr is not None and not sticky)
    if is_masked and thr is not None:
      path = 'masked_confidence_until'
    elif sticky and not is_masked and thr is not None:
      path = 'ucc_confidence_until'
    elif arpc:
      path = 'arpc_ancestral_schedule'
    elif dead_thr:
      path = 'ancestral_dead_thr'  # documented Unif footgun; not a pin bug
    elif thr is None:
      path = 'mid_alpha_ancestral'
    else:
      path = 'UNEXPECTED'
    packing = {
        'hierarchical_kv': sampler.hierarchical_kv,
        'single_stream': sampler.single_stream_decode,
        'sub_block': sampler.sub_block_size,
        'use_block_cache': sampler.use_block_cache,
        'allow_full_seq': sampler.allow_full_seq_decode,
    }
    return path, dead_thr, packing

  INTENT = {
      'baseline': 'floor_confidence alias',
      'hierarchical': 'floor_confidence thr0.9 dual-pack',
      'hierarchical_ancestral': 'forensic mid-α ancestral',
      'hierarchical_ss_ancestral': 'forensic mid-α ancestral + ss',
      'hierarchical_ss': 'floor_confidence + ss',
      'hierarchical_arpc': 'BlockGen ARPC mid-α',
      'hierarchical_quiet': 'ARPC + ss + low T',
      'hubmatch': 'Hub conf thr0.9 ss+sub8',
      'dual_cache': 'Hub DualCache thr1 ss+sub8',
      'full_seq_dual': 'legacy full-seq ancestral',
      'uniform_dual': 'UCC DualCache twin',
      'uniform_commit': 'UCC B1 twin',
      'uniform_commit_t1': 'UCC thr1 B1 pack',
      'uniform_commit_ss': 'UCC hubmatch geometry',
  }

  issues: list[str] = []
  notes: list[str] = []
  results: list[dict] = []

  # H5: baseline == hierarchical
  base = profile_overrides('baseline')
  hier = profile_overrides('hierarchical')
  h5_ok = base == hier
  log('H5', 'baseline_eq_hierarchical', {'equal': h5_ok})
  if not h5_ok:
    issues.append('H5 FAIL: baseline pins != hierarchical')

  # H7: clear_unmask_threshold
  for name, kw in LM_EVAL_DECODE_PROFILES.items():
    thr = kw.get('unmask_threshold', 'MISSING')
    clear = kw.get('clear_unmask_threshold')
    expect_clear = (thr is None)
    ok = ('unmask_threshold' in kw) and (clear is expect_clear)
    log('H7', 'lm_eval_clear_thr', {
        'profile': name, 'thr': thr, 'clear': clear, 'ok': ok})
    if not ok:
      issues.append(f'H7 FAIL {name}: thr={thr} clear={clear}')

  # H6: full_seq_dual pin completeness
  template_keys = {t.split('=')[0] for t in hier}
  fsd_keys = {t.split('=')[0] for t in profile_overrides('full_seq_dual')}
  missing_fsd = sorted(template_keys - fsd_keys)
  log('H6', 'full_seq_dual_missing_pins', {'missing': missing_fsd})
  if missing_fsd:
    issues.append(
        f'H6 WARN full_seq_dual missing vs hierarchical: {missing_fsd}')

  remask_named = {
      'baseline', 'hierarchical', 'hierarchical_ss', 'hubmatch', 'dual_cache',
  }

  warnings.filterwarnings('ignore')
  for name in sorted(DECODE_PROFILES):
    samp = overrides_to_sampling(profile_overrides(name))
    for fp in ('masked', 'uniform'):
      coerced = coerce_profile_for_forward(name, fp)
      cfg = OmegaConf.create({
          'algo': {'forward_process_name': fp, 'hybrid_decode': 'masked'},
          'sampling': samp,
      })
      init_ok = True
      err = None
      path = dead_thr = packing = None
      try:
        sampler = BlockSampler(config=cfg, forward_process=None)
        path, dead_thr, packing = classify(sampler)
      except Exception as e:
        init_ok = False
        err = f'{type(e).__name__}: {e}'

      if init_ok and samp.get('use_arpc') and fp == 'masked':
        ok4 = samp.get('unmask_threshold') is None
        log('H4', 'arpc_thr', {'profile': name, 'ok': ok4})
        if not ok4:
          issues.append(
              f'H4 FAIL {name}/masked: ARPC with thr='
              f'{samp["unmask_threshold"]}')

      if name == 'dual_cache' and init_ok:
        ok3 = (
            sampler.use_block_cache and sampler.hierarchical_kv
            and sampler.single_stream_decode
            and sampler.unmask_threshold == 1.0)
        log('H3', 'dual_cache_pins', {
            'fp': fp, 'ok': ok3, 'path': path, 'packing': packing})
        if not ok3:
          issues.append(f'H3 FAIL dual_cache/{fp} pins inconsistent')

      if init_ok and fp == 'uniform' and name in remask_named and dead_thr:
        # Design note — not a FAIL. Remask thr is MASK-only; Unif twin is UCC.
        notes.append(
            f'H2 NOTE {name}/uniform: thr={sampler.unmask_threshold} '
            f'ignored → {path} (use uniform_commit/UCC for Unif remask)')
        log('H2', 'dead_thr_uniform', {
            'profile': name, 'path': path,
            'thr': sampler.unmask_threshold, 'severity': True})

      if name == 'full_seq_dual' and init_ok:
        if sampler.hierarchical_kv:
          issues.append(
              f'H6 FAIL full_seq_dual/{fp}: hierarchical_kv forced True '
              f'(allow={sampler.allow_full_seq_decode})')
        log('H6', 'full_seq_hkv', {
            'fp': fp, 'hkv': sampler.hierarchical_kv,
            'allow': sampler.allow_full_seq_decode})

      if not init_ok:
        if name.startswith('uniform_') and fp == 'masked':
          h1_ok = 'uniform_confidence_sticky' in (err or '')
          log('H1', 'ucc_masked_refuse', {
              'profile': name, 'ok': h1_ok, 'error': err})
          if not h1_ok:
            issues.append(
                f'H1 FAIL {name}/masked should refuse sticky, got {err}')
          results.append({
              'profile': name, 'fp': fp, 'path': 'REFUSED',
              'match': h1_ok, 'init_ok': False})
        else:
          issues.append(f'INIT FAIL {name}/{fp}: {err}')
          log('INIT', 'sampler_init_fail', {
              'profile': name, 'fp': fp, 'error': err})
        continue

      if name in (
          'hierarchical_ancestral', 'hierarchical_ss_ancestral',
          'full_seq_dual'):
        expected = 'mid_alpha_ancestral'
      elif name in ('hierarchical_arpc', 'hierarchical_quiet'):
        expected = 'arpc_ancestral_schedule'
      elif name.startswith('uniform_'):
        expected = 'ucc_confidence_until'
      elif name in remask_named:
        expected = (
            'masked_confidence_until' if fp == 'masked'
            else 'ancestral_dead_thr')
      else:
        expected = '?'

      match = (path == expected)

      row = {
          'profile': name, 'fp': fp, 'coerced': coerced,
          'path': path, 'dead_thr': dead_thr, 'expected': expected,
          'match': match, 'packing': packing,
          'intent': INTENT.get(name),
          'thr': samp.get('unmask_threshold'),
          'sticky': samp.get('uniform_confidence_sticky', False),
          'arpc': samp.get('use_arpc', False),
          'init_ok': True,
      }
      results.append(row)
      log('PATH', 'profile_path', row)

  for name in DECODE_PROFILES:
    s = overrides_to_sampling(profile_overrides(name))
    log('BAN', 'ban_mask_pin', {
        'profile': name,
        'ban': s.get('ban_mask_pad_logits', 'DEFAULT'),
    })

  print('=== PATH TABLE ===')
  hdr = f'{"profile":28} {"fp":8} {"path":28} {"dead":5} {"ok":5}'
  print(hdr)
  for r in results:
    if not r.get('init_ok', True) and r.get('path') == 'REFUSED':
      print(f'{r["profile"]:28} {r["fp"]:8} {"REFUSED":28} {"-":5} '
            f'{str(r["match"]):5}')
      continue
    if not r.get('init_ok', True):
      continue
    print(
        f'{r["profile"]:28} {r["fp"]:8} {r["path"]:28} '
        f'{str(r["dead_thr"]):5} {str(r["match"]):5}')

  print('\n=== ISSUES ===')
  if not issues:
    print('(none)')
  for i in issues:
    print('-', i)
  print('\n=== NOTES (by design) ===')
  for n in notes:
    print('-', n)
  fails = [i for i in issues if 'FAIL' in i]
  print(f'\nn_profiles={len(DECODE_PROFILES)} n_issues={len(issues)} '
        f'n_notes={len(notes)} n_fails={len(fails)}')
  log('SUM', 'audit_summary', {
      'n_profiles': len(DECODE_PROFILES),
      'n_issues': len(issues),
      'n_notes': len(notes),
      'n_fails': len(fails),
      'issues': issues,
      'notes': notes,
  })
  return 0 if not fails else 1


if __name__ == '__main__':
  raise SystemExit(main())
