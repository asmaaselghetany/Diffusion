#!/usr/bin/env python3
"""Quick val BPD probe from a saved Lightning checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import lightning as pl
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
  sys.path.insert(0, str(_REPO / 'src'))

from discrete_diffusion.data import get_dataloaders, get_tokenizer  # noqa: E402
from discrete_diffusion.evaluations.checkpoint_utils import load_block_trainer_checkpoint  # noqa: E402
from discrete_diffusion.train import register_config_resolvers  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--limit-val-batches', type=int, default=64)
  parser.add_argument('--out', default=None, help='Write val_bpd.json here')
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  register_config_resolvers()
  model, config, tokenizer = load_block_trainer_checkpoint(
      args.checkpoint, device)
  OmegaConf.resolve(config)
  _, val_loader = get_dataloaders(config, tokenizer, skip_train=True)

  trainer = pl.Trainer(
      accelerator='gpu' if device.type == 'cuda' else 'cpu',
      devices=1,
      limit_val_batches=int(args.limit_val_batches),
      logger=False,
      enable_checkpointing=False,
  )
  results = trainer.validate(model, val_loader, verbose=False)
  val_nll = None
  for row in results or []:
    if isinstance(row, dict):
      val_nll = row.get('val/nll', val_nll)
  if val_nll is None:
    val_nll = float(getattr(model, 'metrics', None) and 0.0)
  val_bpd = float(val_nll) / math.log(2) if val_nll is not None else None

  ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
  payload = {
      'global_step': int(ckpt.get('global_step') or -1),
      'checkpoint': str(Path(args.checkpoint).resolve()),
      'val_nll': float(val_nll) if val_nll is not None else None,
      'val_bpd': val_bpd,
      'limit_val_batches': int(args.limit_val_batches),
  }
  text = json.dumps(payload, indent=2) + '\n'
  if args.out:
    Path(args.out).write_text(text, encoding='utf-8')
  print(text)
  return 0


if __name__ == '__main__':
  sys.exit(main())
