"""Shared checkpoint loading for standalone eval / sampling scripts."""

from __future__ import annotations

from pathlib import Path

import hydra.utils
import torch
from omegaconf import OmegaConf

from discrete_diffusion.data import get_tokenizer


def load_block_trainer_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    hydra_overrides: list[str] | None = None,
):
  """Load BlockTrainer ckpt and apply EMA weights when present.

  Optional ``hydra_overrides`` are merged into the saved config before
  instantiate (e.g. ``sampling.hierarchical_kv=true`` for decode-only eval).
  """
  path = Path(checkpoint_path).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(f'checkpoint not found: {path}')
  ckpt = torch.load(path, map_location='cpu', weights_only=False)
  if 'hyper_parameters' not in ckpt or 'config' not in ckpt['hyper_parameters']:
    raise ValueError(f'{path} missing hyper_parameters.config')
  config = ckpt['hyper_parameters']['config']
  if not OmegaConf.is_config(config):
    config = OmegaConf.create(config)
  if hydra_overrides:
    config = merge_hydra_overrides(config, hydra_overrides)
  tokenizer = get_tokenizer(config)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  model = algo_cls.load_from_checkpoint(
      str(path),
      config=config,
      tokenizer=tokenizer,
      map_location=device,
  )
  model.to(device)
  if getattr(model, 'ema', None) is not None:
    model._eval_mode()
  else:
    model.eval()
  return model, config, tokenizer


def merge_hydra_overrides(config, overrides: list[str]):
  """Merge ``key=value`` Hydra tokens into an OmegaConf (eval-time overrides)."""
  OmegaConf.set_struct(config, False)
  for token in overrides:
    if '=' not in token:
      continue
    key, raw = token.split('=', 1)
    raw = raw.strip()
    if raw in ('true', 'True'):
      val = True
    elif raw in ('false', 'False'):
      val = False
    elif raw in ('null', 'None'):
      val = None
    elif raw.startswith('[') or raw.startswith('{'):
      val = OmegaConf.create(raw)
    else:
      try:
        val = int(raw)
      except ValueError:
        try:
          val = float(raw)
        except ValueError:
          val = raw.strip('"').strip("'")
    OmegaConf.update(config, key, val, merge=True)
  return config


__all__ = ['load_block_trainer_checkpoint', 'merge_hydra_overrides']
