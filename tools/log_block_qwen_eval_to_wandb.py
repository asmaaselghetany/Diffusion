#!/usr/bin/env python3
"""Publish block_qwen post-train evals as FOUR clean WandB runs (sections).

  bqwen_gen_ppl   — generative PPL only
  bqwen_elbo      — ELBO sweep only
  bqwen_depbench  — DepBench only (partial OK)
  bqwen_samples   — text samples only

Usage:
  WANDB_MODE=online python tools/log_block_qwen_eval_to_wandb.py \\
    outputs/block_qwen/ar2block_masked_131655 \\
    outputs/block_qwen/ar2block_uniform_133161 \\
    outputs/block_qwen/block_masked_133150 \\
    outputs/block_qwen/block_uniform_133151 \\
    --entity asmaaselghetany-elghitany
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path) -> dict | None:
  if not path.exists():
    return None
  with open(path, encoding='utf-8') as f:
    return json.load(f)


def _parse_arm(run_dir: Path) -> tuple[str, str, str]:
  name = run_dir.name
  m = re.match(r'^(ar2block|block|blockgen)_(masked|uniform)_', name)
  if m:
    line, process = m.group(1), m.group(2)
    if line == 'blockgen':
      line = 'block'
    return f'{line}/{process}', line, process
  arm = name.rsplit('_', 1)[0]
  return f'ar2block/{arm}', 'ar2block', arm


def _mean(vals: list[float]) -> float | None:
  return sum(vals) / len(vals) if vals else None


_SAMPLE_SPLIT = re.compile(r'^Sample\s+(\d+):\s*$', re.MULTILINE)


def _parse_samples_txt(path: Path, max_chars: int = 1200) -> list[dict]:
  if not path.exists():
    return []
  raw = path.read_text(encoding='utf-8', errors='replace')
  matches = list(_SAMPLE_SPLIT.finditer(raw))
  out: list[dict] = []
  for i, m in enumerate(matches):
    start = m.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
    text = raw[start:end].strip()
    out.append({
        'index': int(m.group(1)),
        'text': text[:max_chars] + ('…' if len(text) > max_chars else ''),
        'n_chars': len(text),
    })
  return out


def _collect(run_dir: Path, max_sample_chars: int) -> dict:
  eval_dir = run_dir / 'eval'
  label, line, process = _parse_arm(run_dir)
  out: dict = {
      'arm': label,
      'line': line,
      'process': process,
      'run_dir': str(run_dir),
      'gen': None,
      'elbo_rows': [],
      'dep_overall': None,
      'dep_by_family': [],
      'samples': [],
  }

  gen = _load_json(eval_dir / 'gen_ppl_metrics.json')
  if gen:
    out['gen'] = {
        'arm': label,
        'line': line,
        'process': process,
        'ppl': float(gen['ppl']),
        'acc': float(gen['acc']),
        'avg_nll': float(gen['avg_nll']),
        'median_nll': float(gen['median_nll']),
        'tokens': int(gen['tokens']),
        'pretrained_model': gen.get('pretrained_model'),
    }

  elbo = _load_json(eval_dir / 'block_elbo_sweep.json')
  if elbo and elbo.get('results'):
    for r in sorted(elbo['results'], key=lambda x: int(x['block_size'])):
      out['elbo_rows'].append({
          'arm': label,
          'line': line,
          'process': process,
          'block_size': int(r['block_size']),
          'bpd': float(r['bpd']),
          'nll': float(r['mean_nll']),
          'ppl': float(r['ppl']),
      })

  dep = _load_json(eval_dir / 'depbench_full.json')
  if isinstance(dep, dict) and dep.get('summaries'):
    by_fam: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    for s in dep['summaries']:
      if not isinstance(s, dict):
        continue
      fam = str(s.get('family', 'unknown'))
      for key in ('exact_match_rate', 'violation_rate', 'constraint_rate'):
        if isinstance(s.get(key), (int, float)):
          by_fam[fam][key].append(float(s[key]))
          overall[key].append(float(s[key]))
    out['dep_overall'] = {
        'arm': label,
        'line': line,
        'process': process,
        'exact_match': _mean(overall['exact_match_rate']),
        'violation': _mean(overall['violation_rate']),
        'constraint': _mean(overall['constraint_rate']),
        'n_summaries': len(dep['summaries']),
        'n_results': int(dep.get('num_results') or 0),
    }
    for fam, metrics in sorted(by_fam.items()):
      out['dep_by_family'].append({
          'arm': label,
          'family': fam,
          'exact_match': _mean(metrics['exact_match_rate']),
          'violation': _mean(metrics['violation_rate']),
          'constraint': _mean(metrics['constraint_rate']),
          'n_configs': len(metrics['exact_match_rate']),
      })

  samples = _parse_samples_txt(eval_dir / 'samples.txt', max_chars=max_sample_chars)
  out['samples'] = [{'arm': label, **s} for s in samples]
  return out


def _table(rows: list[dict], cols: list[str]):
  import wandb
  return wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])


def _bar(rows: list[dict], label_key: str, value_key: str, title: str):
  import wandb
  present = [r for r in rows if r.get(value_key) is not None]
  if not present:
    return None
  t = wandb.Table(
      columns=[label_key, value_key],
      data=[[r[label_key], r[value_key]] for r in present],
  )
  return wandb.plot.bar(t, label_key, value_key, title=title)


def _init_run(project: str, entity: str | None, run_id: str, name: str,
              job_type: str, notes: str):
  import wandb
  kw = dict(
      project=project,
      id=run_id,
      name=name,
      resume='allow',
      job_type=job_type,
      notes=notes,
  )
  if entity:
    kw['entity'] = entity
  return wandb.init(**kw)


def publish_gen_ppl(arms: list[dict], project: str, entity: str | None) -> str:
  import wandb
  rows = [a['gen'] for a in arms if a['gen']]
  rid = 'bqwen_gen_ppl'
  run = _init_run(
      project, entity, rid, 'eval · gen-PPL',
      'eval_gen_ppl',
      'Post-train generative perplexity (gpt2-large, first chunk, retokenized).',
  )
  cols = ['arm', 'line', 'process', 'ppl', 'acc', 'avg_nll', 'median_nll',
          'tokens', 'pretrained_model']
  run.log({'table': _table(rows, cols)})
  chart = _bar(rows, 'arm', 'ppl', 'Generative PPL ↓')
  if chart:
    run.log({'bar_ppl': chart})
  chart = _bar(rows, 'arm', 'acc', 'Gen-PPL token accuracy ↑')
  if chart:
    run.log({'bar_acc': chart})
  for r in rows:
    safe = r['arm'].replace('/', '_')
    run.summary[f'ppl/{safe}'] = r['ppl']
    run.summary[f'acc/{safe}'] = r['acc']
  wandb.finish()
  return rid


def publish_elbo(arms: list[dict], project: str, entity: str | None) -> str:
  import wandb
  rows = [r for a in arms for r in a['elbo_rows']]
  rid = 'bqwen_elbo'
  run = _init_run(
      project, entity, rid, 'eval · ELBO',
      'eval_elbo',
      'Validation ELBO sweep vs probe block size (1,4,16,32).',
  )
  run.log({'sweep': _table(
      rows, ['arm', 'line', 'process', 'block_size', 'bpd', 'nll', 'ppl'])})

  # Line chart: X=block_size, one series per arm
  xs = sorted({r['block_size'] for r in rows})
  labels = []
  ys_bpd = []
  ys_nll = []
  for a in arms:
    if not a['elbo_rows']:
      continue
    by_bs = {r['block_size']: r for r in a['elbo_rows']}
    labels.append(a['arm'])
    ys_bpd.append([by_bs[b]['bpd'] for b in xs])
    ys_nll.append([by_bs[b]['nll'] for b in xs])
  if labels:
    run.log({
        'bpd_vs_block_size': wandb.plot.line_series(
            xs=xs, ys=ys_bpd, keys=labels,
            title='ELBO bits/dim vs block size ↓', xname='block_size'),
        'nll_vs_block_size': wandb.plot.line_series(
            xs=xs, ys=ys_nll, keys=labels,
            title='ELBO NLL vs block size ↓', xname='block_size'),
    })

  # Headline @32
  for a in arms:
    r32 = next((r for r in a['elbo_rows'] if r['block_size'] == 32), None)
    if r32:
      safe = a['arm'].replace('/', '_')
      run.summary[f'bpd_bs32/{safe}'] = r32['bpd']
      run.summary[f'nll_bs32/{safe}'] = r32['nll']
  headline = []
  for a in arms:
    r32 = next((r for r in a['elbo_rows'] if r['block_size'] == 32), None)
    if r32:
      headline.append({'arm': a['arm'], 'bpd_bs32': r32['bpd'], 'nll_bs32': r32['nll']})
  chart = _bar(headline, 'arm', 'bpd_bs32', 'ELBO bpd @ block_size=32 ↓')
  if chart:
    run.log({'bar_bpd_bs32': chart})
  wandb.finish()
  return rid


def publish_depbench(arms: list[dict], project: str, entity: str | None) -> str:
  import wandb
  overall = [a['dep_overall'] for a in arms if a['dep_overall']]
  by_fam = [r for a in arms for r in a['dep_by_family']]
  rid = 'bqwen_depbench'
  run = _init_run(
      project, entity, rid, 'eval · DepBench',
      'eval_depbench',
      'DepBench grid means (exact-match / violation / constraint). '
      'Incomplete arms omitted until depbench_full.json exists.',
  )
  if overall:
    run.log({'overall': _table(
        overall,
        ['arm', 'line', 'process', 'exact_match', 'violation', 'constraint',
         'n_summaries', 'n_results'])})
    for key, title in (
        ('exact_match', 'DepBench exact-match ↑'),
        ('violation', 'DepBench violation ↓'),
    ):
      chart = _bar(overall, 'arm', key, title)
      if chart:
        run.log({f'bar_{key}': chart})
    for r in overall:
      safe = r['arm'].replace('/', '_')
      run.summary[f'exact_match/{safe}'] = r['exact_match']
      run.summary[f'violation/{safe}'] = r['violation']
  if by_fam:
    run.log({'by_family': _table(
        by_fam,
        ['arm', 'family', 'exact_match', 'violation', 'constraint', 'n_configs'])})
  pending = [a['arm'] for a in arms if not a['dep_overall']]
  run.summary['arms_done'] = len(overall)
  run.summary['arms_pending'] = len(pending)
  run.notes = (
      (run.notes or '')
      + (f'\nPending: {", ".join(pending)}' if pending else '\nAll arms complete.')
  )
  wandb.finish()
  return rid


def publish_samples(arms: list[dict], project: str, entity: str | None,
                    preview_n: int = 8) -> str:
  import wandb
  rid = 'bqwen_samples'
  run = _init_run(
      project, entity, rid, 'eval · samples',
      'eval_samples',
      'Post-train generated text samples (64 / arm). Open per-arm tables or HTML preview.',
  )
  combined = []
  for a in arms:
    samples = a['samples']
    if not samples:
      continue
    safe = a['arm'].replace('/', '_')
    table = _table(samples, ['arm', 'index', 'n_chars', 'text'])
    run.log({f'table_{safe}': table})
    run.summary[f'n/{safe}'] = len(samples)
    combined.extend(samples)
    # HTML preview
    parts = [f'<h3>{html.escape(a["arm"])} — preview {preview_n}/{len(samples)}</h3>']
    for s in samples[:preview_n]:
      parts.append(
          f'<div style="border:1px solid #ddd;margin:8px 0;padding:8px">'
          f'<b>#{s["index"]}</b> <span style="color:#666">({s["n_chars"]} chars)</span>'
          f'<pre style="white-space:pre-wrap;font-size:12px">'
          f'{html.escape(s["text"])}</pre></div>'
      )
    run.log({f'preview_{safe}': wandb.Html(''.join(parts))})
  if combined:
    run.log({'table_all': _table(combined, ['arm', 'index', 'n_chars', 'text'])})
  wandb.finish()
  return rid


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('run_dirs', nargs='+', type=Path)
  parser.add_argument('--project', default=os.environ.get('WANDB_PROJECT', 'block_qwen'))
  parser.add_argument('--entity', default=os.environ.get('WANDB_ENTITY') or None)
  parser.add_argument('--max-sample-chars', type=int, default=1200)
  parser.add_argument(
      '--sections',
      default='gen_ppl,elbo,depbench,samples',
      help='Comma subset of: gen_ppl,elbo,depbench,samples',
  )
  args = parser.parse_args()

  sections = {s.strip() for s in args.sections.split(',') if s.strip()}
  arms = []
  for run_dir in args.run_dirs:
    run_dir = run_dir.resolve()
    if not (run_dir / 'eval').is_dir():
      print(f'Skip {run_dir}: no eval/')
      continue
    packed = _collect(run_dir, args.max_sample_chars)
    arms.append(packed)
    print(
        f"  {packed['arm']}: gen={packed['gen'] is not None} "
        f"elbo={len(packed['elbo_rows'])} "
        f"dep={packed['dep_overall'] is not None} "
        f"samples={len(packed['samples'])}",
        flush=True,
    )
  if not arms:
    print('Nothing to log.')
    return 1

  entity = args.entity
  project = args.project
  published = []
  if 'gen_ppl' in sections:
    published.append(publish_gen_ppl(arms, project, entity))
  if 'elbo' in sections:
    published.append(publish_elbo(arms, project, entity))
  if 'depbench' in sections:
    published.append(publish_depbench(arms, project, entity))
  if 'samples' in sections:
    published.append(publish_samples(arms, project, entity))

  base = f'https://wandb.ai/{entity or "ENTITY"}/{project}/runs'
  print('\nClean sections:')
  for rid in published:
    print(f'  {base}/{rid}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
