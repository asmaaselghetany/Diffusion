"""Resolve / validate paper levers against configs/levers/registry.yaml."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / 'configs' / 'levers' / 'registry.yaml'


def load_registry(path: Path | None = None) -> dict[str, Any]:
  path = path or DEFAULT_REGISTRY
  with path.open(encoding='utf-8') as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict) or 'levers' not in data:
    raise ValueError(f'Invalid lever registry: {path}')
  return data


def _require_ok(
    selected: set[str],
    requires: list[str],
    mode: str,
) -> bool:
  if not requires:
    return True
  mode = mode or 'all'
  if mode == 'all':
    return set(requires) <= selected
  if mode == 'any':
    return bool(set(requires) & selected)
  raise ValueError(f'Unknown require_mode={mode}')


def resolve(
    *,
    lever_ids: list[str] | None = None,
    preset: str | None = None,
    arm: str,
    line: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Validate lever combo and return Hydra overrides + metadata.

  Raises ValueError on any wiring mistake (wrong arm, conflicts, missing deps).
  """
  reg = registry or load_registry()
  levers_def = reg['levers']
  presets = reg.get('presets') or {}

  selected: list[str] = list(lever_ids or [])
  if preset:
    if preset not in presets:
      raise ValueError(
          f'Unknown preset={preset!r}. Known: {sorted(presets)}')
    p = presets[preset]
    selected = list(p.get('levers') or [])
    allowed_arms = list(p.get('arms') or ['masked', 'uniform'])
    allowed_lines = list(p.get('line') or ['ar2block', 'block'])
    if arm not in allowed_arms:
      raise ValueError(
          f'preset={preset} only allows arms={allowed_arms}, got arm={arm}')
    if line not in allowed_lines:
      raise ValueError(
          f'preset={preset} only allows line={allowed_lines}, got line={line}')

  unknown = [x for x in selected if x not in levers_def]
  if unknown:
    raise ValueError(
        f'Unknown lever id(s) {unknown}. Known: {sorted(levers_def)}')

  selected_set = set(selected)
  overrides: list[str] = []
  sources: list[str] = []
  is_xfer = bool(preset and str(preset).startswith('xfer_'))

  for lid in selected:
    spec = levers_def[lid]
    arms = list(spec.get('arms') or [])
    lines = list(spec.get('line') or ['ar2block', 'block'])
    family = str(spec.get('family') or 'shared')
    if arm not in arms:
      raise ValueError(
          f'lever={lid} only allowed on arms={arms}, got arm={arm}')
    if line not in lines:
      raise ValueError(
          f'lever={lid} only allowed on line={lines}, got line={line}')
    # Family firewall: native BlockGen knobs on ar2block only via xfer_*.
    if family == 'native' and line == 'ar2block' and not is_xfer:
      raise ValueError(
          f'lever={lid} is family=native (BlockGen/scratch). '
          f'Use --line block (B3_/blockgen_*), or an xfer_* preset '
          f'for geometry-on-conversion — do not claim BlockGen on ar2block.')
    if family == 'conversion' and line == 'block':
      raise ValueError(
          f'lever={lid} is family=conversion (AR→block / Fast-dLLM). '
          f'Use --line ar2block; scratch block is the native family.')
    for other in spec.get('conflicts') or []:
      if other in selected_set:
        raise ValueError(
            f'lever conflict: {lid} conflicts with {other}')
    req = list(spec.get('requires') or [])
    mode = str(spec.get('require_mode') or 'all')
    if not _require_ok(selected_set, req, mode):
      raise ValueError(
          f'lever={lid} requires {mode} of {req}; selected={sorted(selected_set)}')
    for tok in spec.get('overrides') or []:
      overrides.append(str(tok))
    sources.append(str(spec.get('source') or lid))

  # Soft cross-check: complementary without shift is allowed but flagged.
  warnings: list[str] = []
  if 'complementary' in selected_set and 'shift' not in selected_set:
    warnings.append(
        'complementary without shift — Fast-dLLM recipe usually pairs both')
  if 'hybrid_p10' in selected_set:
    warnings.append(
        'hybrid_p10 pins default p_uniform=0.1; real effect is --arm hybrid')
  if is_xfer:
    warnings.append(
        'xfer_* = BlockGen geometry on AR→block conversion — '
        'not a BlockGen paper claim (use blockgen_* / B3_* + line=block)')

  tag = preset or ('+'.join(selected) if selected else 'neutral')
  return {
      'arm': arm,
      'line': line,
      'preset': preset,
      'levers': selected,
      'overrides': overrides,
      'tag': tag,
      'sources': sources,
      'warnings': warnings,
  }


_KNOWN_OVERRIDE_PREFIXES = (
    'algo.', 'sampling.', 'trainer.', 'model.', 'loader.', 'eval.',
    'noise.', 'optim.', 'data.', 'wandb.', 'checkpointing.', 'strategy.',
    'callbacks.', 'hydra.',
)

# Hydra config-group switches (no dotted prefix), e.g. data=openwebtext-blockgen.
_KNOWN_GROUP_KEYS = frozenset({
    'data', 'model', 'algo', 'noise', 'sampling', 'strategy', 'lr_scheduler',
    'callbacks', 'prior', 'forward_process',
})

# Bare names that must be prefixed — root-level Hydra keys never merge into algo/sampling.
_BARE_ALGO_HOOKS = frozenset({
    'shift_loss_targets', 'complementary_masks', 'complementary_batching',
    'mask_schedule',
    'joint_ar_alpha', 'hybrid_p_uniform', 'hybrid_decode',
    'block_size_mixture', 'causal_clean_stream', 'stratified_gamma',
    'block_weights', 'block_size_per_gpu', 'pure_noise_block_sizes',
    'loss_type_special_cases', 'single_stream_train', 'hub_struct_attn_only',
    'uniform_simplex_mode',
})
_BARE_SAMPLING_HOOKS = frozenset({
    'hierarchical_kv', 'use_block_cache', 'single_stream_decode',
    'sub_block_size',
    'use_arpc', 'arpc_mode', 'arpc_corruption_mode',
})


def validate_extra_overrides(tokens: list[str]) -> list[str]:
  """Return warnings for EXTRA_OVERRIDES; raise on empty key / bad shape."""
  warnings: list[str] = []
  for tok in tokens:
    if not tok or tok.startswith('#'):
      continue
    if '=' not in tok:
      raise ValueError(
          f'EXTRA_OVERRIDES token must be key=value, got {tok!r}')
    key = tok.split('=', 1)[0].strip()
    if not key:
      raise ValueError(f'EXTRA_OVERRIDES empty key in {tok!r}')
    if not any(key.startswith(p) for p in _KNOWN_OVERRIDE_PREFIXES):
      if key not in _KNOWN_GROUP_KEYS and key not in (
          'block_size', 'scratch_dir'):
        warnings.append(
            f'EXTRA_OVERRIDES key {key!r} has unknown prefix '
            f'(expected one of {_KNOWN_OVERRIDE_PREFIXES[:6]}…)')
    # Orphan-root footgun: bare keys at experiment root never merge.
    if key.startswith('algo.') is False and key in _BARE_ALGO_HOOKS:
      raise ValueError(
          f'EXTRA_OVERRIDES {key!r} looks like an algo hook without '
          f'the algo. prefix — use algo.{key}=...')
    if key.startswith('sampling.') is False and key in _BARE_SAMPLING_HOOKS:
      raise ValueError(
          f'EXTRA_OVERRIDES {key!r} looks like a sampling hook without '
          f'the sampling. prefix — use sampling.{key}=...')
  return warnings


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--preset', default=None)
  p.add_argument(
      '--levers', default='',
      help='Comma-separated lever ids (ignored if --preset set)')
  p.add_argument('--arm', required=True, choices=['masked', 'uniform', 'hybrid'])
  p.add_argument(
      '--line', default='ar2block', choices=['ar2block', 'block', 'blockgen'])
  p.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
  p.add_argument(
      '--extra-overrides', default='',
      help='Space-separated EXTRA_OVERRIDES to validate (optional)')
  p.add_argument(
      '--format', choices=['json', 'overrides', 'export'], default='json',
      help='json metadata | overrides one-per-line | export shell assignments')
  args = p.parse_args(argv)

  lever_ids = [x.strip() for x in args.levers.split(',') if x.strip()]
  line = 'block' if args.line == 'blockgen' else args.line
  try:
    result = resolve(
        lever_ids=None if args.preset else lever_ids,
        preset=args.preset,
        arm=args.arm,
        line=line,
        registry=load_registry(args.registry),
    )
    if args.extra_overrides.strip():
      import shlex
      extra_warn = validate_extra_overrides(shlex.split(args.extra_overrides))
      result.setdefault('warnings', []).extend(extra_warn)
  except ValueError as e:
    print(f'LEVER_ERROR: {e}', file=sys.stderr)
    return 2

  for w in result['warnings']:
    print(f'LEVER_WARN: {w}', file=sys.stderr)

  if args.format == 'json':
    print(json.dumps(result, indent=2))
  elif args.format == 'overrides':
    for tok in result['overrides']:
      print(tok)
  else:
    # Shell-safe export for submit_lever.sh
    ov = ' '.join(result['overrides'])
    print(f"LEVER_TAG={result['tag']}")
    print(f"LEVER_OVERRIDES={ov}")
    print(f"LEVER_LIST={','.join(result['levers'])}")
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
