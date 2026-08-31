#!/usr/bin/env python3
"""Thesis longitudinal eval bundles (PAPER_EXPERIMENTS §7).

Writes standardized artifacts under ``<run_root>/eval/step_<N>/``:

  train_metrics.json, val_bpd.json, gen_ppl.json, throughput.json,
  lm_eval/SUMMARY.json, generation_samples/samples.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
  sys.path.insert(0, str(_REPO / 'src'))


DEFAULT_STEPS = [0, 500, 1000, 2000, 4000, 6000]
LIGHT = {500, 1000}
MEDIUM = {2000, 4000}
FULL = {6000}


def _py() -> str:
  return sys.executable


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = False) -> int:
  print('>>', ' '.join(cmd), flush=True)
  return subprocess.run(cmd, cwd=cwd or _REPO, check=check).returncode


def _tier(step: int) -> str:
  if step in FULL:
    return 'full'
  if step in MEDIUM:
    return 'medium'
  return 'light'


def _tasks_for(step: int) -> str:
  tier = _tier(step)
  if tier == 'full':
    return os.environ.get(
        'MILESTONE_TASKS_FULL',
        'gsm8k,ifeval,mmlu,humaneval')
  if tier == 'medium':
    return os.environ.get('MILESTONE_TASKS_MEDIUM', 'gsm8k,ifeval')
  return os.environ.get('MILESTONE_TASKS_LIGHT', 'gsm8k')


def _find_ckpt(ckpt_dir: Path, step: int) -> Path | None:
  for name in (f'0-{step}.ckpt', f'1-{step}.ckpt'):
    p = ckpt_dir / name
    if p.is_file():
      return p
  best = None
  best_step = -1
  for p in ckpt_dir.glob('*.ckpt'):
    if p.is_symlink():
      continue
    try:
      import torch
      obj = torch.load(p, map_location='cpu', weights_only=False)
      gs = int(obj.get('global_step') or -1)
    except Exception:
      continue
    if gs <= step and gs > best_step:
      best_step = gs
      best = p
  return best


def _write_train_metrics(out: Path, *, step: int, ckpt: Path | None) -> None:
  tokens_per_step = int(os.environ.get('TOKENS_PER_STEP', 256 * 2048))
  payload = {
      'global_step': step,
      'checkpoint': str(ckpt) if ckpt else None,
      'tokens_seen_est': step * tokens_per_step,
  }
  (out / 'train_metrics.json').write_text(
      json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _bundle_step(
    run_root: Path,
    step: int,
    *,
    skip_init: bool,
) -> int:
  out = run_root / 'eval' / f'step_{step}'
  out.mkdir(parents=True, exist_ok=True)
  ckpt_dir = run_root / 'checkpoints'

  ckpt: Path | None
  if step == 0:
    if skip_init:
      print('skip step_0 (--skip-init)')
      return 0
    init_ckpt = ckpt_dir / 'init-step0.ckpt'
    if not init_ckpt.is_file():
      _run([
          _py(), 'tools/materialize_step0_ckpt.py',
          '--out', str(init_ckpt),
          '--line', os.environ.get('LINE', 'ar2block'),
          '--arm', os.environ.get('ARM', 'masked'),
      ])
    ckpt = init_ckpt if init_ckpt.is_file() else None
  else:
    ckpt = _find_ckpt(ckpt_dir, step) if ckpt_dir.is_dir() else None

  if ckpt is None:
    print(f'WARNING: no checkpoint for step {step}; writing metrics only',
          file=sys.stderr)
    _write_train_metrics(out, step=step, ckpt=None)
    return 0

  _write_train_metrics(out, step=step, ckpt=ckpt)

  _run([
      _py(), 'tools/run_val_bpd_probe.py',
      '--checkpoint', str(ckpt),
      '--limit-val-batches', os.environ.get('VAL_BPD_BATCHES', '64'),
      '--out', str(out / 'val_bpd.json'),
  ])

  num_steps = os.environ.get('NUM_STEPS', '32')
  samples_dir = out / 'generation_samples'
  samples_dir.mkdir(parents=True, exist_ok=True)
  samples_pt = samples_dir / 'samples.pt'
  gen_ppl = out / 'gen_ppl.json'
  eval_sub = out / '_eval_tmp'
  eval_sub.mkdir(parents=True, exist_ok=True)
  _run([
      'bash', 'examples/block_qwen/eval.sh',
      str(ckpt), str(eval_sub),
  ], check=False)
  if (eval_sub / 'gen_ppl_metrics.json').is_file():
    shutil.copy2(eval_sub / 'gen_ppl_metrics.json', gen_ppl)
  if (eval_sub / 'samples.pt').is_file():
    shutil.copy2(eval_sub / 'samples.pt', samples_pt)

  lm_dir = out / 'lm_eval'
  lm_dir.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env['OUT_DIR'] = str(lm_dir)
  env['TASKS'] = _tasks_for(step)
  env['NUM_STEPS'] = num_steps
  env['SKIP_THROUGHPUT'] = '0' if _tier(step) == 'full' else '1'
  subprocess.run(
      ['bash', 'examples/block_qwen/lm_eval.sh', str(ckpt)],
      cwd=_REPO, env=env, check=False)

  if _tier(step) == 'full':
    tp = out / 'throughput.json'
    subprocess.run([
        _py(), '-m', 'discrete_diffusion.evaluations.decode_throughput',
        f'checkpoint_path={ckpt}',
        f'metrics_path={tp}',
        f'num_steps={num_steps}',
        'batch_size=1',
        'num_batches=4',
        'mode=conditional',
        'device=cuda',
    ], cwd=_REPO, check=False)
    if not tp.is_file() and (lm_dir / 'tok_s.json').is_file():
      shutil.copy2(lm_dir / 'tok_s.json', tp)

  manifest = {
      'run_root': str(run_root),
      'step': step,
      'tier': _tier(step),
      'checkpoint': str(ckpt),
      'artifacts': {
          'train_metrics': str(out / 'train_metrics.json'),
          'val_bpd': str(out / 'val_bpd.json'),
          'gen_ppl': str(gen_ppl) if gen_ppl.is_file() else None,
          'lm_eval_summary': str(lm_dir / 'SUMMARY.json')
          if (lm_dir / 'SUMMARY.json').is_file() else None,
          'throughput': str(out / 'throughput.json')
          if (out / 'throughput.json').is_file() else None,
      },
  }
  (out / 'milestone_manifest.json').write_text(
      json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
  if eval_sub.exists():
    shutil.rmtree(eval_sub, ignore_errors=True)
  return 0


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--run-root', required=True)
  parser.add_argument('--steps', nargs='+', type=int,
                      default=DEFAULT_STEPS)
  parser.add_argument('--skip-init', action='store_true')
  args = parser.parse_args(argv)

  run_root = Path(args.run_root).resolve()
  if not run_root.is_dir():
    print(f'Missing run_root: {run_root}', file=sys.stderr)
    return 1

  rc = 0
  for step in args.steps:
    print(f'=== milestone step {step} ===')
    rc |= _bundle_step(run_root, step, skip_init=args.skip_init)
  return min(rc, 1)


if __name__ == '__main__':
  sys.exit(main())
