#!/usr/bin/env python3
"""Regression audit: decode-contract drift (scripts/docs vs live profiles).

Checks:
  A — coerce does not silently remap to UCC / hierarchical_ss
  B — hierarchical floor is remask (thr=0.9 greedy), not mid-α ancestral
  C — helper scripts/docs do not advertise old coerce / ancestral floor
  D — operational defaults do not point at contaminated U0 1849335
  E — baseline pins == hierarchical; live profile set present

Exit 0 if clean, 1 if any check fails.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIM_U0 = '1955203'
CONTAMINATED_U0 = '1849335'


def main() -> int:
  sys.path.insert(0, str(ROOT / 'src'))
  from discrete_diffusion.evaluations.decode_profiles import (
      DECODE_PROFILES,
      coerce_profile_for_forward,
      profile_overrides,
  )

  failures: list[str] = []

  # --- A: coerce remaps ---
  cases = [
      ('baseline', 'uniform'),
      ('baseline', 'masked'),
      ('hierarchical', 'uniform'),
      ('hierarchical_ss', 'uniform'),
      ('hubmatch', 'uniform'),
      ('dual_cache', 'uniform'),
      ('uniform_commit', 'uniform'),
  ]
  coerce_map = {
      f'{p}|{fp}': coerce_profile_for_forward(p, fp) for p, fp in cases
  }
  bad_a = {
      k: v for k, v in coerce_map.items()
      if ('hierarchical|' in k or 'hubmatch|' in k or 'dual_cache|' in k)
      and v in ('uniform_commit', 'uniform_dual', 'hierarchical_ss')
      and k.split('|')[0] != v
  }
  if coerce_map.get('baseline|uniform') != 'hierarchical':
    failures.append(
        f"A: baseline|uniform → {coerce_map.get('baseline|uniform')!r} "
        f"(want hierarchical)")
  if bad_a:
    failures.append(f'A: illegal silent remaps: {bad_a}')
  print(f"[A] coerce OK → baseline→hierarchical; remaps={bad_a or '{}'}")

  # --- B: hierarchical pins vs ancestral claim ---
  hier = profile_overrides('hierarchical')
  anc = profile_overrides('hierarchical_ancestral')
  hier_src = (
      ROOT / 'src/discrete_diffusion/evaluations/decode_profiles.py'
  ).read_text(encoding='utf-8')
  m = re.search(
      r"((?:[ \t]*#[^\n]*\n)+)[ \t]*'hierarchical':\s*\[", hier_src)
  comment = m.group(1) if m else ''
  claims_self_ancestral = bool(re.search(
      r'mid-α ancestral|ancestral skeleton|ancestral floor',
      comment, flags=re.I)) and not bool(re.search(
      r'hierarchical_ancestral|forensic only|thr=null is',
      comment, flags=re.I))
  has_thr = any('unmask_threshold=0.9' in t for t in hier)
  has_greedy = any('greedy=true' in t for t in hier)
  anc_null = any('unmask_threshold=null' in t for t in anc)
  if not (has_thr and has_greedy and anc_null):
    failures.append(
        f'B: pin mismatch thr0.9={has_thr} greedy={has_greedy} '
        f'anc_null={anc_null}')
  if claims_self_ancestral:
    failures.append('B: hierarchical comment claims self is ancestral')
  print(
      f'[B] hierarchical remask floor OK '
      f'(thr0.9={has_thr} greedy={has_greedy}; ancestral thr=null)'
  )

  # --- C: stale script comments / notes ---
  stale_patterns = [
      (r'coerces?\s*→\s*hierarchical_ss', 'eval.sh-style ss coerce'),
      (r'coerce to hierarchical_ss', 'ss coerce default'),
      (r'mid-α ancestral skeleton', 'lm_eval ancestral mislabel'),
      (r'BlockSampler ancestral \(steps=32, no ARPC',
       'infer helper ancestral baseline'),
      (r'no unmask_threshold, no DualCache', 'infer helper thr=null claim'),
      (r'uniform coerces → hierarchical_ss', 'eval.sh header'),
      (r'BlockGen block skeleton floor \(both arms\):\s*mid-α ancestral',
       'lm_eval.sbatch ancestral floor'),
      (r'\*\*mid-α ancestral\*\*\s*\(`hierarchical`', 'BASELINE mid-α floor'),
  ]
  scan_files = [
      ROOT / 'examples/block_qwen/eval.sh',
      ROOT / 'scripts/slurm/lm_eval.sbatch',
      ROOT / 'scripts/_infer_block_qwen_eval_profile.bash',
      ROOT / 'scripts/submit_family_eval.sh',
      ROOT / 'docs/research/DEEP_DECODE_FIX_PASS_2026-09-23.md',
      ROOT / 'docs/research/BASELINE.md',
  ]
  hits = []
  for path in scan_files:
    if not path.exists():
      continue
    text = path.read_text(encoding='utf-8', errors='replace')
    for pat, label in stale_patterns:
      for i, line in enumerate(text.splitlines(), 1):
        if re.search(pat, line, flags=re.I):
          hits.append(
              f'{path.relative_to(ROOT)}:{i} [{label}] {line.strip()[:120]}')
  if hits:
    failures.append(f'C: {len(hits)} stale wording hits')
    for h in hits[:20]:
      print(f'  ! {h}')
  print(f'[C] stale wording hits={len(hits)}')

  # --- D: contaminated U0 operational defaults ---
  u0_hits = []
  assign_re = re.compile(
      rf'(CKPT|ckpt|--ckpt)\s*[=:]?\s*[^\n]*{CONTAMINATED_U0}'
      rf'|ar2block_uniform_{CONTAMINATED_U0}'
      rf'|JOB_TAG\s*=\s*{CONTAMINATED_U0}',
      re.I,
  )
  deny_re = re.compile(
      r'contaminat|do not cite|historical|not\s+(?:use|cite)|claim\s+U0',
      re.I,
  )
  for top in ('scripts', 'examples', 'tools'):
    base = ROOT / top
    if not base.is_dir():
      continue
    for path in base.rglob('*'):
      if not path.is_file():
        continue
      if path.suffix not in {'.sh', '.bash', '.sbatch', '.py', '.md'}:
        continue
      if path.name == 'audit_stale_contract.py':
        continue
      try:
        text = path.read_text(encoding='utf-8', errors='replace')
      except OSError:
        continue
      if CONTAMINATED_U0 not in text:
        continue
      for i, line in enumerate(text.splitlines(), 1):
        if CONTAMINATED_U0 not in line:
          continue
        if deny_re.search(line):
          continue
        if assign_re.search(line):
          u0_hits.append(
              f'{path.relative_to(ROOT)}:{i} {line.strip()[:120]}')
  if u0_hits:
    failures.append(
        f'D: {len(u0_hits)} contaminated U0 defaults '
        f'(claim U0={CLAIM_U0})')
    for h in u0_hits[:20]:
      print(f'  ! {h}')
  print(f'[D] contaminated U0 defaults={len(u0_hits)} (claim={CLAIM_U0})')

  # --- E: live contract ---
  baseline_eq = (
      profile_overrides('baseline') == profile_overrides('hierarchical')
  )
  if not baseline_eq:
    failures.append('E: baseline pins != hierarchical')
  required = {
      'baseline', 'hierarchical', 'hierarchical_ancestral',
      'hierarchical_ancestral_t01', 'hierarchical_ss_ancestral',
      'ss_quiet_ancestral', 'hierarchical_arpc_t01',
      'hierarchical_ss', 'uniform_commit', 'dual_cache', 'hubmatch',
  }
  missing = required - set(DECODE_PROFILES)
  if missing:
    failures.append(f'E: missing profiles {sorted(missing)}')
  print(
      f'[E] baseline==hierarchical={baseline_eq}; '
      f'n_profiles={len(DECODE_PROFILES)}'
  )

  if failures:
    print('\nFAIL:')
    for f in failures:
      print(f'  - {f}')
    return 1
  print('\nPASS: decode contract clean')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
