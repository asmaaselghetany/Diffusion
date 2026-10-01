#!/usr/bin/env python3
"""Harvest GSM / NFE / commit-share rows for UCC bake controls (CPU only).

Prefer ``nfe_metrics.json`` (sampler counters written beside ``tok_s_lm_eval.json``).
Older D1/D2 runs lack counters — those rows report wall-clock proxies only.

Usage:
  python tools/harvest_ucc_nfe.py
  python tools/harvest_ucc_nfe.py --root .../ar2block_uniform_2092151 --out ...
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


# Prose alias map (never cite code id as DualCache).
ALIAS = {
    'uniform_commit': 'ucc_thr0.9_b1pack',
    'uniform_dual': 'ucc_thr1_sub8',
    'uniform_dual_random': 'ucc_thr1_sub8_random',
    'ucc_l2r_sub8': 'ucc_thr1_sub8_ltr',
    'ss_quiet_ancestral': 'ss_sub8_T0.1_ancestral',
    'hierarchical_quiet': 'arpc_ss_T0.1',
}

BAKE_DIRS = [
    ('D1', 'uniform_commit', 'lm_eval_bake_20260930_D1_ucommit_uniform_commit_s32'),
    ('D2', 'uniform_dual', 'lm_eval_bake_20260930_D2_udual_uniform_dual_s32'),
    ('D3', 'hierarchical_quiet', 'lm_eval_bake_20260930_D3_quiet_hierarchical_quiet_s32'),
    ('D3r2', 'hierarchical_quiet', 'lm_eval_bake_20261001_D3_quiet_hierarchical_quiet_s32'),
    ('D4', 'ss_quiet_ancestral', 'lm_eval_bake_20261001_D4_ss_quiet_anc_s32'),
    ('D5', 'uniform_dual_random', 'lm_eval_bake_20261001_D5_udual_random_s32'),
    ('L2R', 'ucc_l2r_sub8', 'lm_eval_bake_20261001_L2R_ucc_l2r_sub8_s32'),
]


def _gsm_ife(summary: Path) -> tuple[float | None, float | None]:
  if not summary.is_file():
    return None, None
  o = json.loads(summary.read_text())
  tasks = o.get('tasks') or {}
  gsm = (tasks.get('gsm8k') or {}).get('score')
  ife = (tasks.get('ifeval') or {}).get('score')
  return (
      None if gsm is None else 100.0 * float(gsm),
      None if ife is None else 100.0 * float(ife),
  )


def _load_json(path: Path) -> dict[str, Any] | None:
  if not path.is_file():
    return None
  return json.loads(path.read_text())


def harvest_one(cell: str, profile: str, out_dir: Path) -> dict[str, Any]:
  summary = out_dir / 'SUMMARY.json'
  nfe = _load_json(out_dir / 'nfe_metrics.json')
  tok = _load_json(out_dir / 'tok_s_lm_eval.json')
  gsm, ife = _gsm_ife(summary)
  row: dict[str, Any] = {
      'cell': cell,
      'profile_code': profile,
      'profile_alias': ALIAS.get(profile, profile),
      'out_dir': str(out_dir),
      'has_summary': summary.is_file(),
      'gsm_pct': None if gsm is None else round(gsm, 2),
      'ife_pct': None if ife is None else round(ife, 2),
      'nfe_source': 'missing',
      'n_forwards_mean': None,
      'n_forwards_p10': None,
      'n_forwards_p50': None,
      'n_forwards_p90': None,
      'tokens_per_forward_mean': None,
      'thr_commit_share': None,
      'force_max_commit_share': None,
      'tok_s': None,
      'tokens_generated': None,
      'elapsed_s': None,
      'note': '',
  }
  if nfe:
    row['nfe_source'] = 'sampler_counters'
    for k in (
        'n_forwards_mean', 'n_forwards_p10', 'n_forwards_p50', 'n_forwards_p90',
        'tokens_per_forward_mean', 'thr_commit_share', 'force_max_commit_share'):
      v = nfe.get(k)
      row[k] = None if v is None else float(v)
    if row['force_max_commit_share'] is not None and row['force_max_commit_share'] >= 0.95:
      row['note'] = 'force_max_dominated (≥95% commits); thr≈inert'
  elif tok:
    row['nfe_source'] = 'tok_s_proxy_only'
    row['tok_s'] = tok.get('tok_s')
    row['tokens_generated'] = tok.get('tokens_generated')
    row['elapsed_s'] = tok.get('elapsed_s')
    row['note'] = (
        'No nfe_metrics.json (pre-counter run). '
        'Do not treat elapsed/tok_s as NFE.')
  else:
    row['note'] = 'No SUMMARY/nfe/tok_s yet (queued or failed early)'
  # Provisional GSM from failed D3 log pattern — only if no SUMMARY.
  if cell.startswith('D3') and gsm is None:
    log = out_dir / 'lm_eval.log'
    if log.is_file():
      m = re.search(
          r'\|gsm8k\|[^|]*\|flexible-extract\|[^|]*\|exact_match\|[^|]*\|\s*([0-9.]+)',
          log.read_text(errors='ignore'))
      if m:
        row['gsm_pct'] = round(100.0 * float(m.group(1)), 2)
        row['note'] = (row['note'] + '; GSM from log (provisional)').strip('; ')
  return row


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument(
      '--root',
      type=Path,
      default=Path(
          '/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/'
          'ar2block_uniform_2092151'),
  )
  ap.add_argument(
      '--out',
      type=Path,
      default=None,
      help='CSV path (default: <root>/ucc_nfe_harvest.csv)',
  )
  args = ap.parse_args()
  out = args.out or (args.root / 'ucc_nfe_harvest.csv')
  rows = []
  for cell, profile, dirname in BAKE_DIRS:
    rows.append(harvest_one(cell, profile, args.root / dirname))

  fields = list(rows[0].keys()) if rows else []
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

  # Markdown table to stdout
  print(f'Wrote {out}')
  print('| cell | alias | GSM | IFE | mean NFE | p50 | force-max share | source | note |')
  print('|------|-------|----:|----:|---------:|----:|----------------:|--------|------|')
  for r in rows:
    force = (
        '' if r['force_max_commit_share'] is None
        else f"{100.0 * r['force_max_commit_share']:.1f}%")
    print(
        f"| {r['cell']} | {r['profile_alias']} | "
        f"{'' if r['gsm_pct'] is None else r['gsm_pct']} | "
        f"{'' if r['ife_pct'] is None else r['ife_pct']} | "
        f"{'' if r['n_forwards_mean'] is None else round(r['n_forwards_mean'], 1)} | "
        f"{'' if r['n_forwards_p50'] is None else round(r['n_forwards_p50'], 1)} | "
        f"{force} | "
        f"{r['nfe_source']} | {r['note'][:48]} |"
    )


if __name__ == '__main__':
  main()
