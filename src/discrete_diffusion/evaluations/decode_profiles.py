"""Shared decode profiles + free-sample modes for block_qwen eval.

Single source of truth for:
  - ``DECODE_PROFILES`` (baseline / hierarchical / dual_cache)
  - sample_mode inference (native_free vs conversion_free)
  - samples.meta.json schema / reuse gate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


# Shared ancestral core vs paper overlays. Keep lm-eval / throughput / free-gen
# in sync via this table.
DECODE_PROFILES: dict[str, list[str]] = {
    'baseline': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=false',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
    ],
    'hierarchical': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
    ],
    'dual_cache': [
        'sampling.use_arpc=false',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=true',
        'sampling.single_stream_decode=true',
    ],
}

LM_EVAL_DECODE_PROFILES: dict[str, dict[str, Any]] = {
    'baseline': {
        'use_arpc': False,
        'unmask_threshold': None,
        'clear_unmask_threshold': True,
        'hierarchical_kv': False,
        'use_block_cache': False,
        'single_stream_decode': False,
        'sub_block_size': None,
        'greedy': False,
    },
    'hierarchical': {
        'use_arpc': False,
        'unmask_threshold': None,
        'clear_unmask_threshold': True,
        'hierarchical_kv': True,
        'use_block_cache': False,
        'single_stream_decode': False,
        'sub_block_size': None,
        'greedy': False,
    },
    'dual_cache': {
        'use_arpc': False,
        'hierarchical_kv': True,
        'use_block_cache': True,
        'single_stream_decode': True,
    },
}

_CONVERSION_DATA_HINTS = (
    'nemotron', 'sft', 'instruct', 'chat', 'ultrachat', 'sharegpt',
)
_NATIVE_DATA_HINTS = (
    'openwebtext', 'owt', 'lm1b', 'fineweb',
)

META_SCHEMA_VERSION = 1


def infer_sample_mode(model_config) -> str:
  data = str(OmegaConf.select(model_config, 'data.train') or '').lower()
  tokenizer = str(
      OmegaConf.select(model_config, 'data.tokenizer_name_or_path') or '').lower()
  line = str(
      OmegaConf.select(model_config, 'line')
      or OmegaConf.select(model_config, 'experiment.line')
      or '').lower()
  if any(h in data for h in _CONVERSION_DATA_HINTS) or 'ar2block' in line:
    return 'conversion_free'
  if any(h in data for h in _NATIVE_DATA_HINTS) or line == 'block':
    return 'native_free'
  if 'instruct' in tokenizer:
    return 'conversion_free'
  return 'native_free'


def conversion_prefix_text(user_prompt: str | None = None) -> str:
  if user_prompt:
    return ''  # caller should use apply_chat_template
  return (
      '<|im_start|>system\n'
      'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
      '<|im_end|>\n'
      '<|im_start|>assistant\n'
  )


def conversion_prefix_ids(tokenizer, user_prompt: str | None, device):
  import torch
  if user_prompt:
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': str(user_prompt)}],
        add_generation_prompt=True,
        tokenize=False,
    )
  else:
    text = conversion_prefix_text(None)
  ids = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids']
  return ids.to(device), text


def meta_path_for(samples_path: str | Path) -> Path:
  path = Path(samples_path)
  return path.with_suffix('.meta.json')


def write_samples_meta(samples_path: str | Path, meta: dict) -> Path:
  path = meta_path_for(samples_path)
  payload = dict(meta)
  payload.setdefault('schema_version', META_SCHEMA_VERSION)
  path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
  return path


def read_samples_meta(samples_path: str | Path) -> dict | None:
  path = meta_path_for(samples_path)
  if not path.is_file():
    return None
  try:
    return json.loads(path.read_text(encoding='utf-8'))
  except Exception:
    return None


def samples_reusable(
    samples_path: str | Path,
    *,
    sample_mode: str,
    decode_profile: str,
    checkpoint_path: str | Path,
    force_regen: bool = False,
) -> bool:
  """True iff samples.pt + matching samples.meta.json exist for this request."""
  if force_regen:
    return False
  path = Path(samples_path)
  if not path.is_file():
    return False
  meta = read_samples_meta(path)
  if not meta:
    return False
  try:
    ckpt_a = str(Path(checkpoint_path).expanduser().resolve())
    ckpt_b = str(Path(str(meta.get('checkpoint_path', ''))).expanduser().resolve())
  except Exception:
    return False
  return (
      str(meta.get('sample_mode')) == str(sample_mode)
      and str(meta.get('decode_profile')) == str(decode_profile)
      and ckpt_a == ckpt_b
  )


def infer_mode_from_checkpoint_path(checkpoint_path: str | Path) -> str | None:
  """Best-effort family mode from Hydra config beside a Lightning ckpt."""
  ckpt = Path(checkpoint_path).expanduser().resolve()
  candidates = (
      ckpt.parent.parent / '.hydra' / 'config.yaml',
      ckpt.parent / '.hydra' / 'config.yaml',
      ckpt.parent.parent / 'config.yaml',
  )
  for candidate in candidates:
    if not candidate.is_file():
      continue
    try:
      return infer_sample_mode(OmegaConf.load(candidate))
    except Exception:
      continue
  return None


def should_reuse_samples(
    samples_path: str | Path,
    *,
    checkpoint_path: str | Path,
    sample_mode: str = 'auto',
    decode_profile: str = 'baseline',
    force_regen: bool = False,
) -> bool:
  """Eval-script gate: refuse reuse without matching ``samples.meta.json``.

  ``sample_mode=auto`` resolves via checkpoint Hydra config when possible;
  otherwise requires meta mode in ``{native_free, conversion_free}`` and a
  matching profile + checkpoint path.
  """
  if force_regen:
    return False
  mode = str(sample_mode or 'auto').strip().lower()
  profile = str(decode_profile or 'baseline').strip()
  if mode == 'auto':
    resolved = infer_mode_from_checkpoint_path(checkpoint_path)
    if resolved is not None:
      return samples_reusable(
          samples_path,
          sample_mode=resolved,
          decode_profile=profile,
          checkpoint_path=checkpoint_path,
          force_regen=False,
      )
    path = Path(samples_path)
    if not path.is_file():
      return False
    meta = read_samples_meta(path)
    if not meta:
      return False
    if str(meta.get('sample_mode')) not in ('native_free', 'conversion_free'):
      return False
    try:
      ckpt_a = str(Path(checkpoint_path).expanduser().resolve())
      ckpt_b = str(
          Path(str(meta.get('checkpoint_path', ''))).expanduser().resolve())
    except Exception:
      return False
    return (
        str(meta.get('decode_profile')) == profile and ckpt_a == ckpt_b)
  return samples_reusable(
      samples_path,
      sample_mode=mode,
      decode_profile=profile,
      checkpoint_path=checkpoint_path,
      force_regen=False,
  )


def profile_overrides(profile: str) -> list[str]:
  profile = str(profile or 'baseline').strip()
  if profile == 'keep':
    return []
  if profile not in DECODE_PROFILES:
    raise ValueError(
        f'decode_profile={profile!r} not in '
        f'{sorted(DECODE_PROFILES) + ["keep"]}')
  return list(DECODE_PROFILES[profile])


__all__ = [
    'DECODE_PROFILES',
    'LM_EVAL_DECODE_PROFILES',
    'META_SCHEMA_VERSION',
    'conversion_prefix_ids',
    'conversion_prefix_text',
    'infer_mode_from_checkpoint_path',
    'infer_sample_mode',
    'meta_path_for',
    'profile_overrides',
    'read_samples_meta',
    'samples_reusable',
    'should_reuse_samples',
    'write_samples_meta',
]
