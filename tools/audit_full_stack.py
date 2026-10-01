#!/usr/bin/env python3
"""Static *wiring* audit across training, data, launch, resume, eval paths.

This tool is intentionally **not** a behavioral full-stack test. Default mode
only greps/asserts that expected symbols, scripts, and config hooks exist in
source. A correctly *named* but broken function can still pass static checks.

Layers:
  1. algorithm   — loss/timestep/sampler invariants (pytest when --run-tests)
  2. data_sft    — Nemotron chat template, assistant-only labels, cache safety
  3. model       — vocab projection and Qwen wrapper contracts
  4. eval        — lm_eval harness and uniform-arm guards
  5. launch_ddp  — Slurm/Lightning external-rank wiring
  6. resume_ops  — faithful resume and resource migration guards
  7. codex_parity — themes from codex/block-diffusion-fixes commits
  8. paper_ops   — paper cells, lever submitter, milestone tooling

Usage:
  python tools/audit_full_stack.py              # static wiring only
  python tools/audit_full_stack.py --run-tests  # static + core pytest per layer
  python tools/audit_full_stack.py --run-tests --strict  # + orphaned suite
  python tools/audit_full_stack.py --layer launch_ddp
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parents[1]

STATIC_ONLY_DISCLAIMER = (
    'NOTE: static mode checks source presence/wiring only — not behavior. '
    'Re-run with --run-tests for pytest coverage.'
)


@dataclass
class Check:
  name: str
  ok: bool
  detail: str = ''


@dataclass
class LayerResult:
  layer: str
  checks: list[Check] = field(default_factory=list)

  @property
  def passed(self) -> bool:
    return all(c.ok for c in self.checks)


def _read(path: str | Path) -> str:
  return Path(path).read_text(encoding='utf-8')


def _must_contain(text: str, needle: str, *, label: str) -> Check:
  ok = needle in text
  return Check(label, ok, '' if ok else f'missing {needle!r}')


def _must_not_contain(text: str, needle: str, *, label: str) -> Check:
  ok = needle not in text
  return Check(label, ok, '' if ok else f'forbidden {needle!r} present')


def _must_not_match(text: str, pattern: str, *, label: str) -> Check:
  ok = re.search(pattern, text, re.MULTILINE) is None
  return Check(label, ok, '' if ok else f'forbidden pattern {pattern!r} matched')


def _glob_paths(pattern: str) -> list[Path]:
  return sorted(REPO.glob(pattern))


# ---------------------------------------------------------------------------
# Layer 2 — data / SFT
# ---------------------------------------------------------------------------

def audit_data_sft() -> LayerResult:
  layer = LayerResult('data_sft')
  processing = _read(REPO / 'src/discrete_diffusion/data/processing.py')
  loaders = _read(REPO / 'src/discrete_diffusion/data/loaders.py')
  baseline = _read(REPO / 'src/discrete_diffusion/data/conversion_baseline.py')
  cache_mod = _read(REPO / 'src/discrete_diffusion/data/dataset_cache.py')
  trainer = _read(REPO / 'src/discrete_diffusion/algorithms/block_trainer.py')

  layer.checks.extend([
      _must_contain(processing, 'def _group_block_aligned_sft',
                    label='block-aligned SFT grouper'),
      _must_contain(baseline, 'FAST_DLLM_SFT_CHAT_TEMPLATE',
                    label='Fast-dLLM chat template'),
      _must_contain(loaders, 'install_conversion_chat_template',
                    label='loaders installs conversion chat template'),
      _must_contain(loaders, 'return_assistant_tokens_mask=True',
                    label='assistant token mask in tokenization'),
      _must_contain(loaders, '_group_block_aligned_sft',
                    label='loaders uses block-aligned SFT path'),
      _must_contain(loaders, '_with_race_safe_cache',
                    label='race-safe cache wrapper in loaders'),
      _must_contain(cache_mod, '_save_dataset_cache_atomically',
                    label='atomic dataset cache publish'),
      _must_contain(cache_mod, '_exclusive_dataset_cache_lock',
                    label='exclusive cache build lock'),
      _must_contain(trainer, 'def _batch_valid_tokens',
                    label='assistant-only valid token mask'),
      _must_contain(trainer, 'corruption_mask',
                    label='corruption respects supervision mask'),
  ])
  return layer


# ---------------------------------------------------------------------------
# Layer 3 — model
# ---------------------------------------------------------------------------

def audit_model() -> LayerResult:
  layer = LayerResult('model')
  modeling = _read(REPO / 'src/discrete_diffusion/models/qwen/modeling.py')
  layer.checks.extend([
      _must_contain(modeling, 'hidden[:, :n, :]',
                    label='partial lm_head projection on first n tokens'),
      _must_contain(modeling, 'hidden[:, n:, :]',
                    label='optional clean-stream lm_head slice'),
  ])
  return layer


# ---------------------------------------------------------------------------
# Layer 4 — eval
# ---------------------------------------------------------------------------

def audit_eval() -> LayerResult:
  layer = LayerResult('eval')
  utils = _read(REPO / 'src/discrete_diffusion/evaluations/block_qwen_eval_utils.py')
  harness = _read(REPO / 'src/discrete_diffusion/evaluations/block_qwen_lm_eval.py')
  layer.checks.extend([
      _must_contain(utils, 'def require_masked_likelihood',
                    label='uniform loglikelihood guard'),
      _must_contain(harness, 'require_masked_likelihood',
                    label='harness calls masked-likelihood guard'),
      _must_contain(harness, 'block_qwen_eval_utils',
                    label='shared eval utils module wired'),
  ])
  return layer


# ---------------------------------------------------------------------------
# Layer 5 — launch / DDP
# ---------------------------------------------------------------------------

_TRAIN_SBATCH = [
    'scripts/slurm/ar2block_hybrid.sbatch',
    'scripts/slurm/ar2block_masked.sbatch',
    'scripts/slurm/ar2block_uniform.sbatch',
    'scripts/slurm/block_hybrid.sbatch',
    'scripts/slurm/block_masked.sbatch',
    'scripts/slurm/block_uniform.sbatch',
    'scripts/slurm/blockgen_masked.sbatch',
    'scripts/slurm/blockgen_uniform.sbatch',
    'scripts/slurm/masked.sbatch',
    'scripts/slurm/uniform.sbatch',
]


def audit_launch_ddp() -> LayerResult:
  layer = LayerResult('launch_ddp')
  ddp = _read(REPO / 'scripts/_block_qwen_ddp.bash')
  launch = _read(REPO / 'scripts/_block_qwen_launch.bash')
  ar_sft = _read(REPO / 'scripts/slurm/ar_sft.sbatch')
  lever = _read(REPO / 'scripts/submit_lever.sh')
  common = _read(REPO / 'scripts/slurm/_common.sh')

  layer.checks.extend([
      _must_contain(ddp, 'resolve_block_qwen_ddp_resources',
                    label='DDP resource resolver'),
      _must_contain(ddp, 'append_block_qwen_trainer_overrides',
                    label='trainer override helper'),
      _must_contain(ddp, 'trainer.num_nodes', label='trainer.num_nodes override'),
      _must_contain(ddp, 'trainer.devices', label='trainer.devices per node'),
      _must_contain(ddp, '--ntasks-per-node="${GPUS_PER_NODE}"',
                    label='multi-node external ranks'),
      _must_contain(ddp, '--ntasks="${GPUS_PER_NODE}"',
                    label='single-node explicit ntasks'),
      _must_not_match(
          ddp, r'^\s*srun\b[^\n]*--gpus-per-task=1',
          label='no gpus-per-task=1 in DDP srun'),
      _must_contain(launch, 'source "${_SCRIPT_DIR}/_block_qwen_ddp.bash"',
                    label='launch sources DDP helper'),
      _must_contain(launch, 'run_block_qwen_srun_train',
                    label='launch uses shared srun train'),
      _must_contain(ar_sft, 'source scripts/_block_qwen_ddp.bash',
                    label='ar_sft sources DDP helper'),
      _must_contain(ar_sft, 'append_block_qwen_trainer_overrides',
                    label='ar_sft sets num_nodes'),
      _must_not_contain(ar_sft, '--gpus-per-task=1',
                        label='ar_sft avoids gpus-per-task=1'),
      _must_contain(lever, 'GPUS_PER_NODE', label='lever submitter exports gpus/node'),
      _must_contain(lever, 'NUM_NODES', label='lever submitter exports num_nodes'),
      _must_not_contain(common, 'Lightning subprocess DDP',
                        label='_common.sh does not recommend subprocess DDP'),
  ])

  for rel in _TRAIN_SBATCH:
    text = _read(REPO / rel)
    layer.checks.append(_must_contain(text, 'export NUM_NODES=',
                                      label=f'{rel}: NUM_NODES'))
    layer.checks.append(_must_contain(text, 'export GPUS_PER_NODE=',
                                      label=f'{rel}: GPUS_PER_NODE'))
    layer.checks.append(_must_contain(text, '_block_qwen_launch.bash',
                                      label=f'{rel}: uses launch helper'))

  return layer


# ---------------------------------------------------------------------------
# Layer 6 — resume / ops
# ---------------------------------------------------------------------------

def audit_resume_ops() -> LayerResult:
  layer = LayerResult('resume_ops')
  resume = _read(REPO / 'scripts/resume_block_qwen.sh')
  layer.checks.extend([
      _must_contain(resume, 'trainer.num_nodes/devices are required',
                    label='resume requires recorded world size'),
      _must_contain(resume, 'ALLOW_RESOURCE_MIGRATION',
                    label='resource migration guard'),
      _must_contain(resume, 'export GPUS_PER_NODE',
                    label='resume exports GPUS_PER_NODE'),
      _must_contain(resume, 'export NUM_NODES',
                    label='resume exports NUM_NODES'),
      _must_contain(resume, 'checkpointing.resume_ckpt_path',
                    label='resume pins highest valid ckpt path'),
      _must_contain(resume, 'shlex.split',
                    label='resume merges hydra overrides safely'),
  ])
  return layer


# ---------------------------------------------------------------------------
# Layer 7 — codex parity (commit themes → repo signals)
# ---------------------------------------------------------------------------

def audit_codex_parity() -> LayerResult:
  layer = LayerResult('codex_parity')
  themes: list[tuple[str, Callable[[], Check]]] = [
      ('17754e9 SFT pipeline',
       lambda: _must_contain(_read(REPO / 'src/discrete_diffusion/data/processing.py'),
                             '_group_block_aligned_sft', label='17754e9')),
      ('84b898a Fast-dLLM training/decode',
       lambda: _must_contain(
           _read(REPO / 'src/discrete_diffusion/data/conversion_baseline.py'),
           'FAST_DLLM_SFT_CHAT_TEMPLATE', label='84b898a')),
      ('880cefa worktree-safe eval/resume',
       lambda: Check('880cefa', (REPO / 'scripts/_resolve_block_qwen_run.bash').is_file(),
                     '' if (REPO / 'scripts/_resolve_block_qwen_run.bash').is_file()
                     else 'missing _resolve_block_qwen_run.bash')),
      ('a74c601 cache + staged DDP race-safe',
       lambda: _must_contain(_read(REPO / 'src/discrete_diffusion/data/dataset_cache.py'),
                             '_save_dataset_cache_atomically', label='a74c601')),
      ('4ef69dd external DDP GPU visibility',
       lambda: _must_not_match(
           _read(REPO / 'scripts/_block_qwen_ddp.bash'),
           r'^\s*srun\b[^\n]*--gpus-per-task=1',
           label='4ef69dd')),
      ('65bd846 vocab projection opt',
       lambda: _must_contain(_read(REPO / 'src/discrete_diffusion/models/qwen/modeling.py'),
                             'hidden[:, :n, :]', label='65bd846')),
      ('06bd516 resume + eval hardening',
       lambda: _must_contain(_read(REPO / 'scripts/resume_block_qwen.sh'),
                             'ALLOW_RESOURCE_MIGRATION', label='06bd516')),
  ]
  for _name, fn in themes:
    layer.checks.append(fn())
  return layer


# ---------------------------------------------------------------------------
# Layer 8 — paper / ops
# ---------------------------------------------------------------------------

def audit_paper_ops() -> LayerResult:
  layer = LayerResult('paper_ops')
  cells = REPO / 'configs/paper/cells.yaml'
  lever = REPO / 'scripts/submit_lever.sh'
  milestone = REPO / 'tools/run_milestone_eval.py'
  layer.checks.extend([
      Check('paper cells config', cells.is_file(),
            '' if cells.is_file() else 'missing configs/paper/cells.yaml'),
      Check('lever submitter', lever.is_file(),
            '' if lever.is_file() else 'missing submit_lever.sh'),
      Check('milestone eval tool', milestone.is_file(),
            '' if milestone.is_file() else 'missing run_milestone_eval.py'),
      _must_contain(_read(cells), 'launch:', label='cells define launch paths'),
  ])
  return layer


STATIC_LAYERS: dict[str, Callable[[], LayerResult]] = {
    'data_sft': audit_data_sft,
    'model': audit_model,
    'eval': audit_eval,
    'launch_ddp': audit_launch_ddp,
    'resume_ops': audit_resume_ops,
    'codex_parity': audit_codex_parity,
    'paper_ops': audit_paper_ops,
}

# Core pytest set: fast pre-submit gate.
PYTEST_BY_LAYER: dict[str, list[str]] = {
    'algorithm': [
        'tests/test_audit_fixes.py',
        'tests/test_loss_symmetry.py',
        'tests/test_block_shift_loss.py',
        'tests/test_block_losses.py',
    ],
    'data_sft': [
        'tests/test_sft_pipeline.py',
        'tests/test_data_loaders.py',
        'tests/test_chat_template_and_decode_gaps.py',
    ],
    'model': ['tests/test_qwen_modeling.py'],
    'eval': ['tests/test_block_qwen_lm_eval.py', 'tests/test_gen_ppl_eos.py'],
    'launch_ddp': ['tests/test_milestone_gaps.py', 'tests/test_full_stack_audit.py'],
    'resume_ops': ['tests/test_resume_ops.py'],
    'codex_parity': ['tests/test_full_stack_audit.py'],
    'paper_ops': ['tests/test_paper_cells.py', 'tests/test_lever_registry.py'],
}

# Extra modules included only with --strict (orphans previously outside audit).
PYTEST_STRICT_EXTRA: dict[str, list[str]] = {
    'algorithm': [
        'tests/test_forward_process.py',
        'tests/test_block_mask.py',
        'tests/test_block_forward.py',
        'tests/test_block_elbo.py',
        'tests/test_mask_schedule_elbo.py',
        'tests/test_complementary_masks.py',
        'tests/test_joint_ar_and_hybrid.py',
    ],
    'model': [
        'tests/test_qwen_parity.py',
        'tests/test_dual_cache.py',
    ],
    'eval': [
        'tests/test_block_sampler.py',
        'tests/test_arpc_and_hierarchical.py',
        'tests/test_block_init.py',
    ],
}


def _pytest_paths_for_layer(layer_name: str, *, strict: bool) -> list[str]:
  paths = list(PYTEST_BY_LAYER.get(layer_name, []))
  if strict:
    paths.extend(PYTEST_STRICT_EXTRA.get(layer_name, []))
  # De-dupe while preserving order.
  seen: set[str] = set()
  out: list[str] = []
  for p in paths:
    if p not in seen:
      seen.add(p)
      out.append(p)
  return out


def run_pytest(paths: Iterable[str]) -> Check:
  cmd = [sys.executable, '-m', 'pytest', *paths, '-q', '--tb=no']
  proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
  tail = (proc.stdout + proc.stderr).strip().splitlines()
  summary = tail[-1] if tail else f'exit {proc.returncode}'
  return Check('pytest', proc.returncode == 0, summary)


def run_static(layers: list[str] | None = None) -> list[LayerResult]:
  names = layers or list(STATIC_LAYERS)
  return [STATIC_LAYERS[name]() for name in names if name in STATIC_LAYERS]


def run_audit(
    *,
    layers: list[str] | None = None,
    run_tests: bool = False,
    strict: bool = False,
) -> int:
  selected = layers or list({*STATIC_LAYERS, *PYTEST_BY_LAYER})
  results: list[LayerResult] = run_static([n for n in selected if n in STATIC_LAYERS])

  if run_tests:
    for layer_name in PYTEST_BY_LAYER:
      if layers and layer_name not in layers:
        continue
      rel_paths = _pytest_paths_for_layer(layer_name, strict=strict)
      existing = [str(REPO / p) for p in rel_paths if (REPO / p).is_file()]
      if not existing:
        continue
      lr = LayerResult(layer_name)
      label = 'pytest (strict)' if strict and PYTEST_STRICT_EXTRA.get(layer_name) else 'pytest'
      chk = run_pytest(existing)
      chk.name = label
      lr.checks.append(chk)
      results.append(lr)

  failed = 0
  mode = 'strict' if strict and run_tests else ('tests' if run_tests else 'static')
  print(f'=== Full-stack audit ({mode}) ===')
  print(f'Repo: {REPO}')
  if not run_tests:
    print(STATIC_ONLY_DISCLAIMER)
  print()
  for lr in results:
    status = 'PASS' if lr.passed else 'FAIL'
    print(f'[{status}] {lr.layer}')
    for chk in lr.checks:
      mark = '  ✓' if chk.ok else '  ✗'
      suffix = f' — {chk.detail}' if chk.detail else ''
      print(f'{mark} {chk.name}{suffix}')
      if not chk.ok:
        failed += 1
    print()

  total = sum(len(lr.checks) for lr in results)
  passed = total - failed
  kind = 'wiring checks' if not run_tests else 'checks'
  print(f'Summary: {passed}/{total} {kind} passed across {len(results)} layers')
  return 1 if failed else 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--layer', action='append', dest='layers',
      help='Repeatable layer name (default: all). Examples: launch_ddp, data_sft')
  parser.add_argument(
      '--run-tests', action='store_true',
      help='Also run core pytest modules associated with each layer')
  parser.add_argument(
      '--strict', action='store_true',
      help='With --run-tests, also include orphaned training/decode suites')
  args = parser.parse_args()
  return run_audit(
      layers=args.layers, run_tests=args.run_tests, strict=args.strict)


if __name__ == '__main__':
  raise SystemExit(main())
