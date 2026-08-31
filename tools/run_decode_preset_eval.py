#!/usr/bin/env python3
"""Eval-time decode preset (E_* cells) on an existing conversion checkpoint.

Applies sampling overrides from ``configs/levers/registry.yaml`` without
retraining. Writes tok/s + optional gen-PPL under ``<out_dir>/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _repo() -> Path:
  return Path(__file__).resolve().parents[1]


def _load_resolve():
  spec = importlib.util.spec_from_file_location(
      'resolve_lever', _repo() / 'tools' / 'resolve_lever.py')
  mod = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  sys.modules['resolve_lever'] = mod
  spec.loader.exec_module(mod)
  return mod.load_registry, mod.resolve


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--preset', required=True,
                      help='decode_hierarchical | decode_dual_cache | decode_sub_block')
  parser.add_argument('--arm', default='masked')
  parser.add_argument('--line', default='ar2block')
  parser.add_argument('--out-dir', default=None)
  parser.add_argument('--num-steps', type=int, default=32)
  parser.add_argument('--skip-gen-ppl', action='store_true')
  args = parser.parse_args(argv)

  ckpt = Path(args.checkpoint).resolve()
  if not ckpt.is_file():
    print(f'Missing checkpoint: {ckpt}', file=sys.stderr)
    return 1

  load_registry, resolve = _load_resolve()
  reg = load_registry()
  meta = resolve(
      preset=args.preset, arm=args.arm, line=args.line, registry=reg)
  sampling_ov = [
      o for o in meta['overrides'] if o.startswith('sampling.')]
  if not sampling_ov:
    print(f'Preset {args.preset!r} has no sampling.* overrides', file=sys.stderr)
    return 1

  run_root = ckpt.parent.parent
  out = Path(args.out_dir) if args.out_dir else (
      run_root / 'eval' / f'decode_{args.preset}')
  out.mkdir(parents=True, exist_ok=True)
  repo = _repo()
  py = sys.executable

  manifest = {
      'checkpoint': str(ckpt),
      'preset': args.preset,
      'sampling_overrides': sampling_ov,
      'out_dir': str(out),
  }

  # Throughput with eval-time sampling overrides.
  throughput_path = out / 'throughput.json'
  rc = subprocess.run([
      py, str(repo / 'tools' / '_run_decode_throughput_overrides.py'),
      '--checkpoint', str(ckpt),
      '--metrics-path', str(throughput_path),
      '--num-steps', str(args.num_steps),
      *sum([['--override', o] for o in sampling_ov], []),
  ], cwd=repo, check=False).returncode
  if rc != 0:
    print(f'WARNING: throughput probe exited {rc}', file=sys.stderr)
  manifest['throughput'] = str(throughput_path) if throughput_path.is_file() else None

  if not args.skip_gen_ppl:
    samples = out / 'samples.pt'
    gen_ppl = out / 'gen_ppl.json'
    subprocess.run([
        py, str(repo / 'tools' / '_run_generate_samples_overrides.py'),
        '--checkpoint', str(ckpt),
        '--samples-path', str(samples),
        '--num-samples', '32',
        '--batch-size', '1',
        *sum([['--override', o] for o in sampling_ov], []),
        '--override', f'num_steps={args.num_steps}',
    ], cwd=repo, check=False)
    if samples.is_file():
      subprocess.run([
          py, '-m', 'discrete_diffusion.evaluations.generative_ppl',
          '--config-name=gen_ppl_block_qwen',
          f'samples_path={samples}',
          f'metrics_path={gen_ppl}',
          'pretrained_model=gpt2-large',
          'retokenize=true',
          'first_chunk_only=true',
      ], cwd=repo, check=False)
      if gen_ppl.is_file() and (out / 'gen_ppl_metrics.json').is_file():
        gen_ppl.write_text((out / 'gen_ppl_metrics.json').read_text())
    manifest['gen_ppl'] = str(gen_ppl) if gen_ppl.is_file() else None

  (out / 'decode_manifest.json').write_text(
      json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
  print(json.dumps(manifest, indent=2))
  return 0


if __name__ == '__main__':
  sys.exit(main())
