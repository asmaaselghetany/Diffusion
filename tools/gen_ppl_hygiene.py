#!/usr/bin/env python3
"""GenPPL–entropy hygiene helpers (Unifusion-style).

Rules:
  - Never report GenPPL alone: always (ppl, H_mean[, H_med]).
  - Build a tiny qualitative collapse panel from free-gen samples.
  - Aggregate NFE sweep folders into one table/JSON for Fig 4.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _load_metrics(path: Path) -> dict[str, Any]:
  d = json.loads(path.read_text())
  if not isinstance(d, dict):
    raise ValueError(f'expected object in {path}')
  return d


def pair_from_metrics(d: dict[str, Any]) -> dict[str, Any]:
  """Extract the required GenPPL–entropy pair; raise if incomplete."""
  ppl = d.get('ppl')
  h_mean = d.get('unigram_entropy_mean')
  if ppl is None:
    raise ValueError('metrics missing ppl')
  if h_mean is None:
    raise ValueError(
        'metrics missing unigram_entropy_mean — re-run generative_ppl '
        '(Unifusion hygiene: GenPPL alone is banned)')
  warn = d.get('honesty_warning')
  citeable = d.get('citeable')
  if citeable is None:
    # Infer for older metrics files.
    w = str(warn or '')
    citeable = not (
        'too_few_scored_tokens' in w or 'high_eos_rate_short_span' in w)
  if not citeable:
    raise ValueError(
        f'metrics not citeable (honesty_warning={warn!r}) — do not pair/cite')
  return {
      'ppl': float(ppl),
      'unigram_entropy_mean': float(h_mean),
      'unigram_entropy_median': (
          float(d['unigram_entropy_median'])
          if d.get('unigram_entropy_median') is not None else None),
      'unigram_entropy_std': (
          float(d['unigram_entropy_std'])
          if d.get('unigram_entropy_std') is not None else None),
      'eos_rate': d.get('eos_rate'),
      'full_length_rate': d.get('full_length_rate'),
      'honesty_warning': warn,
      'citeable': bool(citeable),
      'fluency_note': d.get('fluency_note') or (
          'GenPPL+H = collapse hygiene only, not linguistic fluency'),
      'first_chunk_only': d.get('first_chunk_only'),
      'format': (
          f"({float(ppl):.2f}, "
          f"H̄={float(h_mean):.3f}"
          + (
              f", H̃={float(d['unigram_entropy_median']):.3f}"
              if d.get('unigram_entropy_median') is not None else '')
          + ')'),
  }


def write_pair_sidecar(metrics_path: Path, out_path: Path | None = None) -> dict:
  d = _load_metrics(metrics_path)
  pair = pair_from_metrics(d)
  out = out_path or metrics_path.with_name('gen_ppl_pair.json')
  payload = {
      'source_metrics': str(metrics_path),
      'note': 'Unifusion hygiene: always cite GenPPL with unigram entropy',
      **pair,
  }
  out.write_text(json.dumps(payload, indent=2) + '\n')
  return payload


def _parse_samples_txt(path: Path) -> list[str]:
  text = path.read_text(errors='replace')
  chunks = re.split(r'(?m)^Sample\s+\d+:\s*$', text)
  # drop header before Sample 0
  bodies = []
  for ch in chunks[1:]:
    body = re.split(r'(?m)^-{20,}\s*$', ch, maxsplit=1)[0].strip()
    bodies.append(body)
  return bodies


def build_collapse_panel(
    samples_txt: Path,
    metrics_path: Path,
    *,
    out_path: Path,
    n_show: int = 3,
) -> dict:
  """Pick 2–3 samples for appendix when PPL/H disagree or look soft."""
  d = _load_metrics(metrics_path)
  pair = pair_from_metrics(d)
  samples = _parse_samples_txt(samples_txt)
  ppl = pair['ppl']
  h = pair['unigram_entropy_mean']

  # Heuristic bins (same thresholds as generative_ppl honesty gate).
  story = 'balanced'
  if ppl < 80.0 and h < 4.5:
    story = 'low_ppl_low_H_possible_collapse'
  elif ppl < 80.0 and h >= 5.5:
    story = 'low_ppl_healthy_H'
  elif ppl >= 120.0 and h >= 5.5:
    story = 'high_ppl_high_H_noisy'
  elif ppl >= 120.0 and h < 5.0:
    story = 'high_ppl_soft_H'

  # Show first, mid, last non-empty bodies (stable, no per-sample H needed).
  nonempty = [(i, s) for i, s in enumerate(samples) if s.strip()]
  picks: list[dict[str, Any]] = []
  if nonempty:
    idxs = {0, len(nonempty) // 2, len(nonempty) - 1}
    for rank, j in enumerate(sorted(idxs)[:n_show]):
      i, s = nonempty[j]
      picks.append({
          'rank': rank,
          'sample_index': i,
          'chars': len(s),
          'preview': s[:600],
      })

  panel = {
      'story': story,
      'pair': pair,
      'n_samples_parsed': len(samples),
      'picks': picks,
      'note': (
          'Appendix qualitative panel (Unifusion-style). Not a claim surface. '
          'Prefer cases where PPL looks good but H is soft, or vice versa.'),
  }
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(panel, indent=2) + '\n')
  # Also a short markdown for thesis appendix paste.
  md = out_path.with_suffix('.md')
  lines = [
      f'# Collapse / hygiene panel',
      f'',
      f'- Story: `{story}`',
      f'- Pair: `{pair["format"]}`',
      f'- Honesty: `{pair.get("honesty_warning")}`',
      f'',
  ]
  for p in picks:
    lines.append(f'## Sample {p["sample_index"]}')
    lines.append('')
    lines.append('```')
    lines.append(p['preview'])
    lines.append('```')
    lines.append('')
  md.write_text('\n'.join(lines))
  return panel


def aggregate_nfe_sweep(sweep_root: Path, out_path: Path | None = None) -> dict:
  """Collect steps_*/gen_ppl_pair.json (or metrics) into one table."""
  rows = []
  for sub in sorted(sweep_root.glob('steps_*')):
    if not sub.is_dir():
      continue
    m = re.match(r'steps_(\d+)', sub.name)
    if not m:
      continue
    steps = int(m.group(1))
    pair_path = sub / 'gen_ppl_pair.json'
    metrics_path = sub / 'gen_ppl_metrics.json'
    if pair_path.is_file():
      pair = json.loads(pair_path.read_text())
    elif metrics_path.is_file():
      pair = pair_from_metrics(_load_metrics(metrics_path))
    else:
      continue
    rows.append({
        'num_steps': steps,
        'dir': sub.name,
        'ppl': pair['ppl'],
        'unigram_entropy_mean': pair['unigram_entropy_mean'],
        'unigram_entropy_median': pair.get('unigram_entropy_median'),
        'format': pair.get('format'),
        'honesty_warning': pair.get('honesty_warning'),
    })
  rows.sort(key=lambda r: r['num_steps'])
  payload = {
      'sweep_root': str(sweep_root),
      'protocol': 'block ancestral NFE sweep; GenPPL reported only with H',
      'rows': rows,
  }
  out = out_path or (sweep_root / 'nfe_gen_ppl_entropy_table.json')
  out.write_text(json.dumps(payload, indent=2) + '\n')
  # CSV for plotting
  csv = out.with_suffix('.csv')
  csv.write_text(
      'num_steps,ppl,unigram_entropy_mean,unigram_entropy_median\n'
      + '\n'.join(
          f"{r['num_steps']},{r['ppl']},{r['unigram_entropy_mean']},"
          f"{r.get('unigram_entropy_median') or ''}"
          for r in rows)
      + '\n')
  return payload


def aggregate_multiseed(root: Path, out_path: Path | None = None) -> dict:
  seeds = []
  for sub in sorted(root.glob('seed_*')):
    if not sub.is_dir():
      continue
    m = re.match(r'seed_(\d+)', sub.name)
    if not m:
      continue
    metrics = sub / 'gen_ppl_metrics.json'
    if not metrics.is_file():
      continue
    pair = pair_from_metrics(_load_metrics(metrics))
    seeds.append({'seed': int(m.group(1)), **pair})
  if not seeds:
    raise FileNotFoundError(f'no seed_*/gen_ppl_metrics.json under {root}')
  ppls = [s['ppl'] for s in seeds]
  hs = [s['unigram_entropy_mean'] for s in seeds]

  def _mean_std(xs: list[float]) -> tuple[float, float]:
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
      return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return mu, math.sqrt(var)

  ppl_mu, ppl_sd = _mean_std(ppls)
  h_mu, h_sd = _mean_std(hs)
  payload = {
      'root': str(root),
      'n_seeds': len(seeds),
      'seeds': seeds,
      'ppl_mean': ppl_mu,
      'ppl_std': ppl_sd,
      'unigram_entropy_mean_of_means': h_mu,
      'unigram_entropy_std_across_seeds': h_sd,
      'format': f'({ppl_mu:.2f}±{ppl_sd:.2f}, H̄={h_mu:.3f}±{h_sd:.3f})',
  }
  out = out_path or (root / 'multiseed_gen_ppl_pair.json')
  out.write_text(json.dumps(payload, indent=2) + '\n')
  return payload


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest='cmd', required=True)

  sp = sub.add_parser('pair', help='Write gen_ppl_pair.json from metrics')
  sp.add_argument('metrics', type=Path)
  sp.add_argument('--out', type=Path, default=None)

  sc = sub.add_parser('collapse-panel', help='Appendix sample panel')
  sc.add_argument('--samples-txt', type=Path, required=True)
  sc.add_argument('--metrics', type=Path, required=True)
  sc.add_argument('--out', type=Path, required=True)
  sc.add_argument('--n-show', type=int, default=3)

  sn = sub.add_parser('aggregate-nfe', help='Aggregate NFE sweep pairs')
  sn.add_argument('sweep_root', type=Path)
  sn.add_argument('--out', type=Path, default=None)

  sm = sub.add_parser('aggregate-multiseed', help='Aggregate seed_* pairs')
  sm.add_argument('root', type=Path)
  sm.add_argument('--out', type=Path, default=None)

  args = p.parse_args(argv)
  if args.cmd == 'pair':
    payload = write_pair_sidecar(args.metrics, args.out)
    print(payload['format'])
  elif args.cmd == 'collapse-panel':
    panel = build_collapse_panel(
        args.samples_txt, args.metrics, out_path=args.out, n_show=args.n_show)
    print(panel['story'], panel['pair']['format'])
  elif args.cmd == 'aggregate-nfe':
    payload = aggregate_nfe_sweep(args.sweep_root, args.out)
    for r in payload['rows']:
      print(f"steps={r['num_steps']}: {r['format']}")
  elif args.cmd == 'aggregate-multiseed':
    payload = aggregate_multiseed(args.root, args.out)
    print(payload['format'])
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
