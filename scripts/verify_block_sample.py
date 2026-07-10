#!/usr/bin/env python3
"""G6: block-wise generation smoke for masked and uniform BlockTrainer. Requires CUDA."""

from __future__ import annotations

import argparse
import os
import sys

import torch


def _run(
    algo: str,
    hub: str,
    length: int,
    block_size: int,
    steps: int,
) -> int:
  config_dir = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

  from hydra import compose, initialize_config_dir
  from hydra.core.global_hydra import GlobalHydra
  from discrete_diffusion.train import register_config_resolvers
  from discrete_diffusion.data import get_tokenizer

  register_config_resolvers()
  overrides = [
      'data=synthetic',
      'model=qwen_block',
      f'algo={algo}',
      'noise=log-linear',
      f'block_size={block_size}',
      f'model.length={length}',
      f'model.hub_id={hub}',
      f'data.tokenizer_name_or_path={hub}',
      f'sampling.steps={steps}',
      'sampling=block',
      'eval.generate_samples=false',
      'trainer.accelerator=cuda',
  ]

  GlobalHydra.instance().clear()
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    config = compose(config_name='config', overrides=overrides)

  from discrete_diffusion.algorithms.block_trainer import BlockTrainer
  from discrete_diffusion.sampling.block_sampler import BlockSampler

  tokenizer = get_tokenizer(config)
  model = BlockTrainer(config, tokenizer=tokenizer)
  model.eval()
  device = torch.device('cuda')
  model = model.to(device)

  sampler = BlockSampler(config)
  samples = sampler.generate(
      model, num_samples=1, num_steps=steps, eps=1e-3, inject_bos=True)

  expected = (1, length)
  if samples.shape != expected:
    print(f'FAIL G6 ({algo}): shape {samples.shape} != {expected}')
    return 1
  if not torch.isfinite(samples.float()).all():
    print(f'FAIL G6 ({algo}): non-finite token ids')
    return 1
  if samples.min() < 0 or samples.max() >= model.vocab_size:
    print(f'FAIL G6 ({algo}): token id out of range')
    return 1
  print(f'G6 PASS ({algo}): shape={tuple(samples.shape)} '
        f'len={samples.shape[1]} steps={steps}')
  return 0


def main() -> int:
  if not torch.cuda.is_available():
    print('FAIL: CUDA required. Run on a GPU node (e.g. via Slurm).')
    return 1

  parser = argparse.ArgumentParser()
  parser.add_argument('--hub', default='Qwen/Qwen2.5-0.5B')
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--block-size', type=int, default=16)
  parser.add_argument('--steps', type=int, default=8)
  parser.add_argument('--algo', choices=['block_masked', 'block_uniform', 'both'],
                      default='both')
  args = parser.parse_args()

  algos = (
      ['block_masked', 'block_uniform'] if args.algo == 'both'
      else [args.algo])
  for algo in algos:
    code = _run(
        algo, args.hub, args.length, args.block_size, args.steps)
    if code != 0:
      return code
  return 0


if __name__ == '__main__':
  sys.exit(main())
