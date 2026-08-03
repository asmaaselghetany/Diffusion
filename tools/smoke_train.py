#!/usr/bin/env python3
"""G0/G3/G4 smoke training — BlockTrainer (masked | uniform). Requires CUDA."""

from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> int:
  if not torch.cuda.is_available():
    print('FAIL: CUDA required. Run on a GPU node (e.g. via Slurm).')
    return 1

  parser = argparse.ArgumentParser()
  parser.add_argument('--algo', choices=['block_masked', 'block_uniform'], required=True)
  parser.add_argument('--steps', type=int, default=1)
  parser.add_argument('--block-size', type=int, default=16)
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--hub', default='Qwen/Qwen2.5-1.5B-Instruct')
  parser.add_argument('--batch-size', type=int, default=2)
  args = parser.parse_args()

  overrides = [
      'data=synthetic',
      'model=qwen_block',
      f'algo={args.algo}',
      'noise=log-linear',
      f'block_size={args.block_size}',
      f'model.length={args.length}',
      f'model.hub_id={args.hub}',
      f'data.tokenizer_name_or_path={args.hub}',
      f'trainer.max_steps={args.steps}',
      'trainer.devices=1',
      'trainer.accumulate_grad_batches=1',
      'trainer.accelerator=cuda',
      f'loader.batch_size={args.batch_size}',
      f'loader.global_batch_size={args.batch_size}',
      f'loader.eval_global_batch_size={args.batch_size}',
      'loader.num_workers=2',
      'loader.pin_memory=true',
      'strategy=single-device',
      'eval.generate_samples=false',
      '++wandb=null',
      'callbacks=[]',
      'checkpointing.resume_from_ckpt=false',
      'trainer.num_sanity_val_steps=0',
      'trainer.limit_val_batches=0',
      'trainer.precision=32',
  ]

  config_dir = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

  from hydra import compose, initialize_config_dir
  from hydra.core.global_hydra import GlobalHydra
  from discrete_diffusion.train import train, register_config_resolvers

  register_config_resolvers()

  GlobalHydra.instance().clear()
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name='config', overrides=overrides)
    train(cfg)

  print(f'Smoke PASS: algo={args.algo}, steps={args.steps}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
