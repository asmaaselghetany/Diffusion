#!/usr/bin/env python3
"""Block_qwen eval orchestrator.

Prefer the upstream-only path first::

  bash examples/block_qwen/eval.sh <checkpoint.ckpt>

That uses UNI-D² ``generate_samples`` + ``generative_ppl`` only.

This script wraps those same modules and optionally adds the block ELBO
sweep. DepBench is no longer part of this pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


_REPO_SRC = _repo_root() / 'src'
if str(_REPO_SRC) not in sys.path:
  sys.path.insert(0, str(_REPO_SRC))


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> int:
  print('>>', ' '.join(cmd), flush=True)
  result = subprocess.run(cmd, cwd=cwd, check=False, env=env)
  if check and result.returncode != 0:
    raise subprocess.CalledProcessError(result.returncode, cmd)
  return result.returncode


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description=(
          'Run block_qwen eval (generate_samples + gen-PPL + optional ELBO)'))
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument(
      '--run-dir', default=None,
      help='Output directory (default: <ckpt_parent>/../eval)')
  parser.add_argument('--num-samples', type=int, default=64)
  # seq 2048 × Qwen-1.5B block sampling OOMs at batch 16 on H100 80GB;
  # keep 1 (same constraint as training eval_global_batch_size).
  parser.add_argument('--gen-batch-size', type=int, default=1)
  parser.add_argument('--gen-steps', type=int, default=32)
  parser.add_argument(
      '--sample-mode', default='auto',
      help='auto|native_free|conversion_free|bare_bos (see generate_samples.yaml)')
  parser.add_argument(
      '--decode-profile', default='baseline',
      help='baseline|hierarchical|hubmatch|dual_cache|keep — free-gen defaults to baseline clears')
  parser.add_argument(
      '--max-new-tokens', type=int, default=512,
      help='Cap free-gen length (default 512; was null→2048 fill)')
  parser.add_argument(
      '--force-regen', action='store_true',
      help='Ignore existing samples.pt even if meta matches')
  parser.add_argument('--skip-samples', action='store_true')
  parser.add_argument('--skip-gen-ppl', action='store_true')
  parser.add_argument(
      '--upstream-only',
      action='store_true',
      help='Only UNI-D² generate_samples + generative_ppl (skip ELBO).')
  parser.add_argument('--skip-elbo', action='store_true')
  # Deprecated no-ops kept so old CLI / sbatch args do not crash.
  parser.add_argument('--skip-depbench', action='store_true',
                      help=argparse.SUPPRESS)
  parser.add_argument('--depbench-count', type=int, default=0,
                      help=argparse.SUPPRESS)
  parser.add_argument('--depbench-root', default=None,
                      help=argparse.SUPPRESS)
  parser.add_argument('--elbo-max-batches', type=int, default=50)
  parser.add_argument('--block-sizes', default='1,4,16,32')
  args = parser.parse_args(argv)

  if args.upstream_only:
    args.skip_elbo = True
  if args.skip_depbench or args.depbench_root or args.depbench_count:
    print('NOTE: DepBench removed from pipeline; ignoring depbench flags.',
          file=sys.stderr)

  repo = _repo_root()
  checkpoint = Path(args.checkpoint).resolve()
  if not checkpoint.exists():
    print(f'Checkpoint not found: {checkpoint}', file=sys.stderr)
    return 1

  run_dir = (
      Path(args.run_dir) if args.run_dir
      else checkpoint.parent.parent / 'eval')
  run_dir.mkdir(parents=True, exist_ok=True)

  samples_path = run_dir / 'samples.pt'
  gen_ppl_path = run_dir / 'gen_ppl_metrics.json'
  elbo_path = run_dir / 'block_elbo_sweep.json'
  manifest_path = run_dir / 'eval_manifest.json'

  py = sys.executable

  if not args.skip_samples:
    from discrete_diffusion.evaluations.decode_profiles import should_reuse_samples
    reuse = should_reuse_samples(
        samples_path,
        checkpoint_path=checkpoint,
        sample_mode=args.sample_mode,
        decode_profile=args.decode_profile,
        force_regen=args.force_regen,
    )
    if reuse:
      print(f'Reusing samples (meta gate matched): {samples_path}')
    else:
      if samples_path.exists():
        print(f'Regenerating samples (stale/missing meta or --force-regen)')
      _run([
          py, '-m', 'discrete_diffusion.evaluations.generate_samples',
          f'checkpoint_path={checkpoint}',
          f'samples_path={samples_path}',
          f'num_samples={args.num_samples}',
          f'batch_size={args.gen_batch_size}',
          f'num_steps={args.gen_steps}',
          f'sample_mode={args.sample_mode}',
          f'decode_profile={args.decode_profile}',
          f'max_new_tokens={args.max_new_tokens}',
          'save_text=true',
          'device=cuda',
      ], cwd=repo)

  if not args.skip_gen_ppl:
    # UNI-D² generative_ppl with block_qwen tokenizer defaults.
    rc = _run([
        py, '-m', 'discrete_diffusion.evaluations.generative_ppl',
        '--config-name=gen_ppl_block_qwen',
        f'samples_path={samples_path}',
        f'metrics_path={gen_ppl_path}',
        'pretrained_model=gpt2-large',
        'retokenize=true',
        'first_chunk_only=true',
    ], cwd=repo, check=False)
    if rc != 0:
      print(f'WARNING: gen-PPL exited {rc}; continuing.', file=sys.stderr)

  if not args.skip_elbo:
    rc = _run([
        py, str(repo / 'tools' / 'run_block_elbo_sweep.py'),
        '--checkpoint', str(checkpoint),
        '--block-sizes', args.block_sizes,
        '--max-batches', str(args.elbo_max_batches),
        '--output', str(elbo_path),
    ], cwd=repo, check=False)
    if rc != 0:
      print(f'WARNING: ELBO sweep exited {rc}; continuing.', file=sys.stderr)

  manifest = {
      'checkpoint': str(checkpoint),
      'run_dir': str(run_dir),
      'samples': str(samples_path) if samples_path.exists() else None,
      'gen_ppl': str(gen_ppl_path) if gen_ppl_path.exists() else None,
      'elbo_sweep': str(elbo_path) if elbo_path.exists() else None,
  }
  with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
  print(json.dumps(manifest, indent=2))
  print(f'Manifest: {manifest_path}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
