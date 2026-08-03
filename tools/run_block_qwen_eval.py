#!/usr/bin/env python3
"""End-to-end block_qwen eval: samples → gen-PPL → DepBench → ELBO sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
  import os
  return Path(os.environ.get(
      'ASMAA_WORKSPACE',
      '/fast/project/HFMI_SynergyUnit/asmaa.elsayed'))


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
  parser = argparse.ArgumentParser(description='Run block_qwen eval suite')
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--run-dir', default=None,
                      help='Output directory (default: <ckpt_parent>/../eval)')
  parser.add_argument('--num-samples', type=int, default=64)
  # seq 2048 × Qwen-1.5B block sampling OOMs at batch 16 on H100 80GB;
  # keep 1 (same constraint as training eval_global_batch_size).
  parser.add_argument('--gen-batch-size', type=int, default=1)
  parser.add_argument('--gen-steps', type=int, default=32)
  parser.add_argument('--skip-samples', action='store_true')
  parser.add_argument('--skip-gen-ppl', action='store_true')
  parser.add_argument('--skip-depbench', action='store_true')
  parser.add_argument('--skip-elbo', action='store_true')
  parser.add_argument('--depbench-count', type=int, default=16)
  parser.add_argument(
      '--depbench-root',
      default=None,
      help='Path to depbench repo root (default: $DEPBENCH_ROOT or '
           '$ASMAA_WORKSPACE/projects/depbench)')
  parser.add_argument('--elbo-max-batches', type=int, default=50)
  parser.add_argument('--block-sizes', default='1,4,16,32')
  args = parser.parse_args(argv)

  repo = _repo_root()
  workspace = _workspace_root()
  checkpoint = Path(args.checkpoint).resolve()
  if not checkpoint.exists():
    print(f'Checkpoint not found: {checkpoint}', file=sys.stderr)
    return 1

  run_dir = Path(args.run_dir) if args.run_dir else checkpoint.parent.parent / 'eval'
  run_dir.mkdir(parents=True, exist_ok=True)

  samples_path = run_dir / 'samples.pt'
  gen_ppl_path = run_dir / 'gen_ppl_metrics.json'
  elbo_path = run_dir / 'block_elbo_sweep.json'
  depbench_path = run_dir / 'depbench_full.json'
  manifest_path = run_dir / 'eval_manifest.json'

  py = sys.executable

  if not args.skip_samples:
    if samples_path.exists():
      print(f'Samples already exist at {samples_path}; skipping generation.')
    else:
      _run([
        py, '-m', 'discrete_diffusion.evaluations.generate_samples',
        f'checkpoint_path={checkpoint}',
        f'samples_path={samples_path}',
        f'num_samples={args.num_samples}',
        f'batch_size={args.gen_batch_size}',
        f'num_steps={args.gen_steps}',
        'save_text=true',
        'device=cuda',
      ], cwd=repo)

  if not args.skip_gen_ppl:
    import torch
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['hyper_parameters']['config']
    tokenizer_name = str(getattr(config.data, 'tokenizer_name_or_path', 'gpt2'))
    rc = _run([
        py, '-m', 'discrete_diffusion.evaluations.generative_ppl',
        f'samples_path={samples_path}',
        f'model_tokenizer={tokenizer_name}',
        'pretrained_model=gpt2-large',
        f'metrics_path={gen_ppl_path}',
        'retokenize=true',
        'first_chunk_only=true',
    ], cwd=repo, check=False)
    if rc != 0:
      print(f'WARNING: gen-PPL exited {rc}; continuing to DepBench/ELBO.',
            file=sys.stderr)

  if not args.skip_elbo:
    rc = _run([
        py, str(repo / 'tools' / 'run_block_elbo_sweep.py'),
        '--checkpoint', str(checkpoint),
        '--block-sizes', args.block_sizes,
        '--max-batches', str(args.elbo_max_batches),
        '--output', str(elbo_path),
    ], cwd=repo, check=False)
    if rc != 0:
      print(f'WARNING: ELBO sweep exited {rc}; continuing to DepBench.',
            file=sys.stderr)

  if not args.skip_depbench:
    import os
    depbench_root = Path(
        args.depbench_root
        or os.environ.get('DEPBENCH_ROOT', '')
        or (workspace / 'projects' / 'depbench')
    ).resolve()
    depbench_script = depbench_root / 'tools' / 'run_depbench.py'
    if not depbench_script.exists():
      print(
          f'DepBench script not found at {depbench_script}. '
          'Set --depbench-root or DEPBENCH_ROOT.',
          file=sys.stderr,
      )
      return 1
    import os as _os
    depbench_env = _os.environ.copy()
    # Make `import depbench` work without editable install (pyproject's
    # uni-d2 file: URL resolves to the wrong path on this cluster layout).
    prev = depbench_env.get('PYTHONPATH', '')
    depbench_env['PYTHONPATH'] = (
        f'{depbench_root}{(":" + prev) if prev else ""}'
    )
    depbench_env['DEPBENCH_ROOT'] = str(depbench_root)
    rc = _run([
        py, str(depbench_script),
        '--checkpoint', str(checkpoint),
        '--families',
        'arithmetic,agreement,coreference,csp,'
        'history_binding,history_tool_arg,'
        'naturalistic_gsm8k,naturalistic_coref',
        '--num-steps', '8,16,32',
        '--tokens-per-step', '1,2,4,8',
        '--block-sizes', '8,16,32',
        '--count', str(args.depbench_count),
        '--batch-size', '1',
        '--plot',
        '--output', str(depbench_path),
    ], cwd=depbench_root, check=False, env=depbench_env)
    if rc != 0:
      print(f'WARNING: DepBench exited {rc}; writing partial manifest.',
            file=sys.stderr)

  manifest = {
      'checkpoint': str(checkpoint),
      'run_dir': str(run_dir),
      'samples': str(samples_path) if samples_path.exists() else None,
      'gen_ppl': str(gen_ppl_path) if gen_ppl_path.exists() else None,
      'elbo_sweep': str(elbo_path) if elbo_path.exists() else None,
      'depbench': str(depbench_path) if depbench_path.exists() else None,
  }
  with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
  print(json.dumps(manifest, indent=2))
  print(f'Manifest: {manifest_path}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
