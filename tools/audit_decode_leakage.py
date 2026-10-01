#!/usr/bin/env python3
"""Offline Part-1 helpers that need existing bake outputs (CPU, no GPU).

  # Spot-check extractor / reasoning → number (checklist item 6)
  python tools/audit_decode_leakage.py extractor \\
    --out-dir .../lm_eval_bake_..._D2_... --n 20

  # Print whether nfe_metrics / SUMMARY exist (readiness for Part-2)
  python tools/audit_decode_leakage.py status --root .../ar2block_uniform_2092151

Does **not** replace the CPU unit probes in tests/test_decode_leakage.py
(items 1/3/4/5). Corrupt-GT accuracy A/B on a live job still needs a
dedicated eval submit that scrambles references before scoring.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_NUM = re.compile(r'(-?\d+(?:\.\d+)?)')


def _load_from_log(out_dir: Path, n: int) -> list[dict]:
  """Parse ``question:`` / ``answer:`` dumps from lm_eval.log."""
  log = out_dir / 'lm_eval.log'
  if not log.is_file():
    raise FileNotFoundError(log)
  text = log.read_text(errors='ignore')
  parts = list(re.finditer(r'=== task=gsm8k[^\n]*===\n', text))
  start = parts[-1].end() if parts else 0
  chunk = text[start:]
  samples: list[dict] = []
  for block in re.finditer(
      r'={10,}\nquestion:\s*(.*?)\nanswer:\s*(.*?)\n={10,}',
      chunk, flags=re.S):
    q, a = block.group(1).strip(), block.group(2).strip()
    samples.append({'doc': q, 'completion': a, 'target': None})
    if len(samples) >= n:
      break
  if not samples:
    raise FileNotFoundError(
        f'No GSM question/answer dumps in {log}')
  return samples


def _load_samples(out_dir: Path) -> list[dict]:
  for name in (
      'samples_gsm8k_flexible-extract.json',
      'samples_gsm8k.json',
      'gsm8k_samples.json',
  ):
    p = out_dir / name
    if p.is_file():
      return json.loads(p.read_text())
  hits = sorted(out_dir.glob('samples*gsm*'))
  if hits:
    return json.loads(hits[0].read_text())
  return _load_from_log(out_dir, n=64)


def cmd_extractor(args: argparse.Namespace) -> None:
  samples = _load_samples(Path(args.samples) if args.samples else Path(args.out_dir))
  n = min(int(args.n), len(samples))
  print(f'Reading {n}/{len(samples)} samples from bake out_dir')
  print('Check by hand: does the reasoning lead to the boxed/final number?')
  print('Flag: number appears with no derivation, or answer leaked in prompt.\n')
  for i, s in enumerate(samples[:n]):
    # lm_eval sample shapes vary; be defensive.
    doc = s.get('doc') or s.get('arguments') or {}
    if isinstance(doc, list):
      prompt = str(doc[0]) if doc else ''
    elif isinstance(doc, dict):
      prompt = str(doc.get('question') or doc.get('query') or doc)
    else:
      prompt = str(doc or s.get('input') or s.get('prompt') or '')
    resp = str(
        s.get('completion')
        or s.get('filtered_resps')
        or s.get('resps')
        or s.get('output')
        or '')
    if isinstance(s.get('filtered_resps'), list) and s['filtered_resps']:
      resp = str(s['filtered_resps'][0])
    elif isinstance(s.get('resps'), list) and s['resps']:
      r0 = s['resps'][0]
      resp = str(r0[0] if isinstance(r0, list) and r0 else r0)
    gold = s.get('target') or s.get('doc', {}).get('answer') if isinstance(
        s.get('doc'), dict) else s.get('target')
    nums = _NUM.findall(resp.replace(',', ''))
    print(f'--- sample {i} ---')
    print('prompt_tail:', prompt.replace('\\n', '\n')[-240:].strip())
    print('response_tail:', resp.replace('\\n', '\n')[-400:].strip())
    print('extracted_nums_tail:', nums[-5:] if nums else [])
    print('gold:', gold)
    # Crude: gold number also sitting in the prompt (few-shot leak).
    if gold is not None and str(gold).split()[-1:] and str(gold) in prompt:
      print('WARN: gold string appears inside prompt text')
    print()


def cmd_status(args: argparse.Namespace) -> None:
  root = Path(args.root)
  # Inline map so this tool runs without package import tricks.
  bake = [
      ('D1', 'uniform_commit', 'lm_eval_bake_20260930_D1_ucommit_uniform_commit_s32'),
      ('D2', 'uniform_dual', 'lm_eval_bake_20260930_D2_udual_uniform_dual_s32'),
      ('D3', 'hierarchical_quiet', 'lm_eval_bake_20260930_D3_quiet_hierarchical_quiet_s32'),
      ('D3r2', 'hierarchical_quiet', 'lm_eval_bake_20261001_D3_quiet_hierarchical_quiet_s32'),
      ('D4', 'ss_quiet_ancestral', 'lm_eval_bake_20261001_D4_ss_quiet_anc_s32'),
      ('D5', 'uniform_dual_random', 'lm_eval_bake_20261001_D5_udual_random_s32'),
      ('L2R', 'ucc_l2r_sub8', 'lm_eval_bake_20261001_L2R_ucc_l2r_sub8_s32'),
  ]
  print(f'root={root}')
  for cell, profile, dirname in bake:
    d = root / dirname
    summ = (d / 'SUMMARY.json').is_file()
    nfe = (d / 'nfe_metrics.json').is_file()
    toks = (d / 'tok_s_lm_eval.json').is_file()
    print(
        f'{cell:5} {profile:22} summary={int(summ)} nfe={int(nfe)} '
        f'tok_s={int(toks)}  {d.name}')


def main() -> None:
  ap = argparse.ArgumentParser()
  sp = ap.add_subparsers(dest='cmd', required=True)
  p1 = sp.add_parser('extractor', help='Hand-audit GSM generations')
  p1.add_argument('--out-dir', required=True)
  p1.add_argument('--samples', default=None, help='Override samples JSON path')
  p1.add_argument('-n', type=int, default=20)
  p1.set_defaults(func=cmd_extractor)
  p2 = sp.add_parser('status', help='Which bake cells have NFE counters')
  p2.add_argument(
      '--root',
      default=(
          '/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/'
          'ar2block_uniform_2092151'),
  )
  p2.set_defaults(func=cmd_status)
  args = ap.parse_args()
  args.func(args)


if __name__ == '__main__':
  main()
