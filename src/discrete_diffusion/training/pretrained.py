"""Load BD3-LM / MDLM pretraining checkpoints into JEDi trainers."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

# Official BD3-LM OWT MDLM pretrain (850k steps) for block-diffusion finetuning.
DEFAULT_BD3LM_OWT_PRETRAIN_HF = 'kuleshov-group/bd3lm-owt-block_size1024-pretrain'
DEFAULT_BD3LM_OWT_PRETRAIN_CKPT = 'bd3lm_owt_block1024_pretrain.ckpt'

# BD3-LM OWT AR baseline (no EOS injection) for AR → block-diffusion init.
DEFAULT_AR_OWT_PRETRAIN_HF = 'kuleshov-group/ar-noeos-owt'
DEFAULT_AR_OWT_PRETRAIN_CKPT = 'ar_noeos_owt.ckpt'


def default_pretrain_ckpt_path(cache_dir: str) -> str:
  return os.path.join(cache_dir, 'checkpoints', DEFAULT_BD3LM_OWT_PRETRAIN_CKPT)


def default_ar_pretrain_ckpt_path(cache_dir: str) -> str:
  return os.path.join(cache_dir, 'checkpoints', DEFAULT_AR_OWT_PRETRAIN_CKPT)


def is_hf_hub_model_id(source: str) -> bool:
  """True for Hugging Face repo ids like ``org/name`` (not local paths)."""
  if not source or os.path.exists(source):
    return False
  if source.endswith(('.ckpt', '.pt', '.pth', '.safetensors')):
    return False
  return '/' in source and '://' not in source


def resolve_pretrained_source(
    finetune_path: str = '',
    from_pretrained: str = '',
    cache_dir: str = '',
) -> str:
  """Resolve training init source from config fields."""
  if from_pretrained and str(from_pretrained).lower() not in ('', 'false', '0', 'none'):
    key = str(from_pretrained).lower()
    if key in ('true', '1', 'default', 'bd3lm', 'mdlm', 'owt'):
      if cache_dir:
        return default_pretrain_ckpt_path(cache_dir)
      return DEFAULT_BD3LM_OWT_PRETRAIN_HF
    if key in ('ar', 'ar-noeos', 'ar_owt', 'ar-owt'):
      if cache_dir:
        return default_ar_pretrain_ckpt_path(cache_dir)
      return DEFAULT_AR_OWT_PRETRAIN_HF
    return str(from_pretrained)
  if finetune_path:
    return str(finetune_path)
  return ''


def load_pretrained_state_dict(source: str) -> dict[str, torch.Tensor]:
  """Load a state dict from a local checkpoint or Hugging Face hub id."""
  if is_hf_hub_model_id(source):
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(source, trust_remote_code=True)
    return dict(model.state_dict())

  if not os.path.isfile(source):
    raise FileNotFoundError(
      f'Pretrained checkpoint not found: {source}. '
      'On Jupiter login, run download_mdlm_pretrain.sh or download_ar_pretrain.sh'
    )

  checkpoint = torch.load(source, map_location='cpu', weights_only=False)
  if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    return checkpoint['state_dict']
  if isinstance(checkpoint, dict):
    return checkpoint
  raise ValueError(f'Unsupported checkpoint format at {source}')


def load_matching_weights(
    model: torch.nn.Module,
    source: str,
    logger: logging.Logger | None = None,
) -> tuple[int, int, list[str]]:
  """Load compatible weights from *source* into *model* (non-strict)."""
  log = logger or logging.getLogger(__name__)
  ckpt_state = load_pretrained_state_dict(source)
  model_state = model.state_dict()

  filtered_state: dict[str, torch.Tensor] = {}
  skipped_keys: list[str] = []
  for key, value in ckpt_state.items():
    if key not in model_state:
      continue
    if model_state[key].shape == value.shape:
      filtered_state[key] = value
    else:
      skipped_keys.append(
        f'{key}: ckpt{list(value.shape)} vs model{list(model_state[key].shape)}'
      )

  model.load_state_dict(filtered_state, strict=False)
  log.info(
    'Loaded pretrained weights from %s: %d/%d checkpoint keys (%d shape skips)',
    source,
    len(filtered_state),
    len(ckpt_state),
    len(skipped_keys),
  )
  if skipped_keys:
    log.info('Skipped %d keys due to shape mismatch:', len(skipped_keys))
    for line in skipped_keys[:8]:
      log.info('  %s', line)
    if len(skipped_keys) > 8:
      log.info('  ... and %d more', len(skipped_keys) - 8)
  return len(filtered_state), len(ckpt_state), skipped_keys


def save_pretrained_ckpt(
    hf_repo: str,
    output_path: str,
    logger: logging.Logger | None = None,
) -> str:
  """Download HF BD3-LM pretrain and save as a Lightning-compatible .ckpt."""
  log = logger or logging.getLogger(__name__)
  os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
  state_dict = load_pretrained_state_dict(hf_repo)
  torch.save({'state_dict': state_dict}, output_path)
  log.info('Saved %d keys to %s', len(state_dict), output_path)
  return output_path
