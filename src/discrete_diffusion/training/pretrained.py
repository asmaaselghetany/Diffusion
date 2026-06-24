"""Load BD3-LM / MDLM pretraining checkpoints into block-diffusion trainers."""

from __future__ import annotations

import logging
import os
from typing import Literal

import omegaconf
import torch

# Official BD3-LM OWT MDLM pretrain (850k steps) for block-diffusion finetuning.
DEFAULT_BD3LM_OWT_PRETRAIN_HF = 'kuleshov-group/bd3lm-owt-block_size1024-pretrain'
DEFAULT_BD3LM_OWT_PRETRAIN_CKPT = 'bd3lm_owt_block1024_pretrain.ckpt'

# BD3-LM OWT AR baseline (no EOS injection) for AR → block-diffusion init.
DEFAULT_AR_OWT_PRETRAIN_HF = 'kuleshov-group/ar-noeos-owt'
DEFAULT_AR_OWT_PRETRAIN_CKPT = 'ar_noeos_owt.ckpt'

PretrainProfile = Literal['ar', 'bd3lm', 'block_diffusion']

_KEY_PREFIX_STRIPS = ('module.', 'model.', 'bd3lm.')


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


def infer_pretrain_profile(
    source: str,
    *,
    from_pretrained: str = '',
    state_dict: dict[str, torch.Tensor] | None = None,
) -> PretrainProfile:
  """Infer backbone layout from config sentinels, path, or checkpoint keys."""
  key = str(from_pretrained).lower()
  if key in ('ar', 'ar-noeos', 'ar_owt', 'ar-owt'):
    return 'ar'
  if key in ('true', '1', 'default', 'bd3lm', 'mdlm', 'owt'):
    return 'bd3lm'

  source_l = source.lower()
  basename = os.path.basename(source_l)
  if 'ar_noeos' in basename or basename.startswith('ar_'):
    return 'ar'
  if 'bd3lm' in source_l or 'mdlm' in source_l:
    return 'bd3lm'

  if state_dict is not None:
    keys = list(state_dict.keys())
    if any('adaLN_modulation' in k or 'sigma_map' in k for k in keys):
      return 'bd3lm'
    return 'ar'

  return 'ar'


def apply_pretrain_profile(config, profile: PretrainProfile) -> None:
  """Align model/algo flags with the checkpoint family before model construction."""
  omegaconf.OmegaConf.set_struct(config, False)
  if hasattr(config, 'model'):
    omegaconf.OmegaConf.set_struct(config.model, False)
  if hasattr(config, 'algo'):
    omegaconf.OmegaConf.set_struct(config.algo, False)

  if profile == 'ar':
    config.model.causal_attention = True
    config.model.adaln = False
    config.algo.cross_attn = False
  elif profile == 'bd3lm':
    config.model.causal_attention = False
    config.model.adaln = True
    config.algo.cross_attn = False
  elif profile == 'block_diffusion':
    config.model.causal_attention = False
    config.model.adaln = False
    config.algo.cross_attn = True
  else:
    raise ValueError(f'Unknown pretrain profile: {profile!r}')

  if hasattr(config, 'algo'):
    omegaconf.OmegaConf.set_struct(config.algo, True)
  if hasattr(config, 'model'):
    omegaconf.OmegaConf.set_struct(config.model, True)
  omegaconf.OmegaConf.set_struct(config, True)


def configure_pretrain_init(
    config,
    *,
    pretrain_source: str,
    logger: logging.Logger | None = None,
) -> dict[str, torch.Tensor] | None:
  """Resolve profile, mutate config, and optionally peek checkpoint keys."""
  log = logger or logging.getLogger(__name__)
  profile = str(
    omegaconf.OmegaConf.select(config, 'training.pretrain_profile', default='auto')
  ).lower()
  from_pretrained = str(
    omegaconf.OmegaConf.select(config, 'training.from_pretrained', default='') or ''
  )

  state_dict: dict[str, torch.Tensor] | None = None
  if profile == 'auto':
    state_dict = load_pretrained_state_dict(pretrain_source)
    profile = infer_pretrain_profile(
      pretrain_source,
      from_pretrained=from_pretrained,
      state_dict=state_dict,
    )

  apply_pretrain_profile(config, profile)  # type: ignore[arg-type]
  log.info(
    'Applied pretrain profile %r for source %s',
    profile,
    pretrain_source,
  )
  return state_dict


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


def _strip_key_prefixes(key: str) -> str:
  stripped = key
  changed = True
  while changed:
    changed = False
    for prefix in _KEY_PREFIX_STRIPS:
      if stripped.startswith(prefix):
        stripped = stripped[len(prefix):]
        changed = True
  return stripped


def _candidate_model_keys(ckpt_key: str) -> list[str]:
  stripped = _strip_key_prefixes(ckpt_key)
  candidates = [ckpt_key, stripped]
  if not stripped.startswith('backbone.'):
    candidates.append(f'backbone.{stripped}')
  return list(dict.fromkeys(candidates))


def _align_checkpoint_to_model(
    ckpt_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
  """Map checkpoint keys onto model keys, keeping the first shape match."""
  filtered_state: dict[str, torch.Tensor] = {}
  skipped_keys: list[str] = []
  for key, value in ckpt_state.items():
    matched = False
    for candidate in _candidate_model_keys(key):
      if candidate not in model_state:
        continue
      if model_state[candidate].shape != value.shape:
        skipped_keys.append(
          f'{key} -> {candidate}: ckpt{list(value.shape)} '
          f'vs model{list(model_state[candidate].shape)}'
        )
        continue
      if candidate not in filtered_state:
        filtered_state[candidate] = value
      matched = True
      break
    if not matched and key in model_state:
      if model_state[key].shape != value.shape:
        skipped_keys.append(
          f'{key}: ckpt{list(value.shape)} vs model{list(model_state[key].shape)}'
        )
  return filtered_state, skipped_keys


def load_matching_weights(
    model: torch.nn.Module,
    source: str,
    logger: logging.Logger | None = None,
    *,
    min_load_fraction: float = 0.0,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[int, int, list[str]]:
  """Load compatible weights from *source* into *model* (non-strict)."""
  log = logger or logging.getLogger(__name__)
  ckpt_state = state_dict if state_dict is not None else load_pretrained_state_dict(source)
  model_state = model.state_dict()

  filtered_state, skipped_keys = _align_checkpoint_to_model(ckpt_state, model_state)

  model.load_state_dict(filtered_state, strict=False)
  loaded = len(filtered_state)
  total = len(ckpt_state)
  load_fraction = loaded / max(total, 1)
  log.info(
    'Loaded pretrained weights from %s: %d/%d checkpoint keys (%.1f%%, %d shape skips)',
    source,
    loaded,
    total,
    100.0 * load_fraction,
    len(skipped_keys),
  )
  if skipped_keys:
    log.info('Skipped %d keys due to shape mismatch:', len(skipped_keys))
    for line in skipped_keys[:8]:
      log.info('  %s', line)
    if len(skipped_keys) > 8:
      log.info('  ... and %d more', len(skipped_keys) - 8)

  if min_load_fraction > 0.0 and load_fraction < min_load_fraction:
    raise RuntimeError(
      f'Pretrained load too low for {source}: loaded {loaded}/{total} keys '
      f'({load_fraction:.1%}) < required {min_load_fraction:.1%}. '
      'Check pretrain_profile (ar vs bd3lm), model=block_dit, and checkpoint path.'
    )
  return loaded, total, skipped_keys


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
