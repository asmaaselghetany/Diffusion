"""Shared checkpoint loading for standalone eval / sampling scripts."""

from __future__ import annotations

from pathlib import Path

import hydra.utils
import torch
from omegaconf import OmegaConf

from discrete_diffusion.data import get_tokenizer
from discrete_diffusion.data.conversion_baseline import (
    apply_checkpoint_embed_vocab_size,
    checkpoint_embed_vocab_size,
)


def trusted_torch_load(path, map_location='cpu'):
  """Load a checkpoint dict from a trusted operator path.

  Tries ``weights_only=True`` first (safe for pure tensor state_dicts). Full
  Lightning ckpts embed OmegaConf / custom objects and require pickle —
  those loads are **trusted-cluster-only** (do not point this at untrusted
  uploads). Prefer safetensors for public weight exchange.
  """
  path = str(path)
  try:
    return torch.load(path, map_location=map_location, weights_only=True)
  except Exception:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_block_trainer_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    hydra_overrides: list[str] | None = None,
):
  """Load BlockTrainer ckpt and apply EMA weights when present.

  Optional ``hydra_overrides`` are merged into the saved config before
  instantiate (e.g. ``sampling.hierarchical_kv=true`` for decode-only eval).

  Vocab: new trains keep Hub-padded ``151936``; old ckpts (``151666``) are
  detected from embed weight shape so load does not size-mismatch.
  """
  path = Path(checkpoint_path).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(f'checkpoint not found: {path}')
  ckpt = trusted_torch_load(path, map_location='cpu')
  if 'hyper_parameters' not in ckpt or 'config' not in ckpt['hyper_parameters']:
    raise ValueError(f'{path} missing hyper_parameters.config')
  config = ckpt['hyper_parameters']['config']
  if not OmegaConf.is_config(config):
    config = OmegaConf.create(config)
  if hydra_overrides:
    config = merge_hydra_overrides(config, hydra_overrides)
  tokenizer = get_tokenizer(config)
  embed_v = checkpoint_embed_vocab_size(ckpt)
  if embed_v is not None:
    apply_checkpoint_embed_vocab_size(tokenizer, embed_v)
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


__all__ = [
    'load_block_trainer_checkpoint',
    'merge_hydra_overrides',
    'trusted_torch_load',
]
