#!/usr/bin/env python3
"""Materialize a global_step=0 block_qwen checkpoint for init (step_0) eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as pl

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
  sys.path.insert(0, str(_REPO / 'src'))

from discrete_diffusion.data import get_tokenizer  # noqa: E402
from discrete_diffusion.train import register_config_resolvers  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from hydra.utils import get_class  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def _compose(algo: str, line: str):
  register_config_resolvers()
  cfg_dir = str(_REPO / 'configs')
  if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
  with initialize_config_dir(config_dir=cfg_dir, version_base='1.3'):
    cfg = compose(
        config_name='config',
        overrides=[
            '+experiment=block_qwen',
            f'algo={algo}',
            f'model.load_pretrained={"true" if line == "ar2block" else "false"}',
        ],
    )
  OmegaConf.resolve(cfg)
  return cfg


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', required=True, help='Output .ckpt path')
  parser.add_argument('--line', default='ar2block', choices=['ar2block', 'block'])
  parser.add_argument('--arm', default='masked',
                      choices=['masked', 'uniform', 'hybrid'])
  args = parser.parse_args()

  algo_map = {
      'masked': 'block_masked',
      'uniform': 'block_uniform',
      'hybrid': 'block_hybrid',
  }
  config = _compose(algo_map[args.arm], args.line)
  tokenizer = get_tokenizer(config)
  algo_cls = get_class(config.algo._target_)
  model = algo_cls(config, tokenizer)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  trainer = pl.Trainer(
      accelerator='cpu',
      devices=1,
      logger=False,
      enable_checkpointing=False,
  )
  trainer.strategy.connect(model)
  trainer.save_checkpoint(str(out))
  # Patch global_step for milestone pickers.
  import torch
  obj = torch.load(out, map_location='cpu', weights_only=False)
  obj['global_step'] = 0
  obj['epoch'] = 0
  torch.save(obj, out)
  print(f'Wrote init checkpoint: {out}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
