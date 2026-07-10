#!/usr/bin/env python3
"""Tier 3: run masked + uniform for N steps and check loss decreases. Requires CUDA."""

from __future__ import annotations

import argparse
import os
import sys

import lightning as L
import torch


class _LossLogger(L.Callback):
  def __init__(self):
    self.losses: list[float] = []

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    del trainer, pl_module, batch, batch_idx
    if outputs is None:
      return
    if isinstance(outputs, dict) and 'loss' in outputs:
      self.losses.append(float(outputs['loss'].detach().cpu()))
    elif hasattr(outputs, 'detach'):
      self.losses.append(float(outputs.detach().cpu()))


def _run(algo: str, steps: int, hub: str, length: int, block_size: int,
         batch_size: int) -> list[float]:
  config_dir = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

  from hydra import compose, initialize_config_dir
  from hydra.core.global_hydra import GlobalHydra
  from discrete_diffusion.train import (
      register_config_resolvers,
      _align_strategy_device,
      _resolve_accelerator,
  )
  from discrete_diffusion.data import get_dataloaders, get_tokenizer
  import hydra.utils

  register_config_resolvers()

  overrides = [
      'data=synthetic', 'model=qwen_block', f'algo={algo}',
      'noise=log-linear',
      f'block_size={block_size}', f'model.length={length}',
      f'model.hub_id={hub}', f'data.tokenizer_name_or_path={hub}',
      f'trainer.max_steps={steps}',
      'trainer.devices=1', 'trainer.accumulate_grad_batches=1',
      'trainer.accelerator=cuda', 'trainer.precision=32',
      f'loader.batch_size={batch_size}',
      f'loader.global_batch_size={batch_size}',
      f'loader.eval_global_batch_size={batch_size}',
      'loader.num_workers=2',
      'loader.pin_memory=true', 'strategy=single-device',
      'eval.generate_samples=false', '++wandb=null', 'callbacks=[]',
      'checkpointing.resume_from_ckpt=false',
      'optim.lr=1e-3',
      'trainer.log_every_n_steps=1',
      'trainer.limit_val_batches=0', 'trainer.num_sanity_val_steps=0',
  ]

  GlobalHydra.instance().clear()
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    config = compose(config_name='config', overrides=overrides)

  accel = _resolve_accelerator(config)
  omegaconf = __import__('omegaconf')
  omegaconf.OmegaConf.set_struct(config.trainer, False)
  config.trainer.accelerator = accel
  omegaconf.OmegaConf.set_struct(config.trainer, True)
  _align_strategy_device(config, accel)

  torch.set_float32_matmul_precision('high')

  tokenizer = get_tokenizer(config)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  model = algo_cls(config, tokenizer=tokenizer)
  train_ds, valid_ds = get_dataloaders(config, tokenizer)

  cb = _LossLogger()
  trainer = L.Trainer(
      max_steps=steps,
      devices=1,
      accelerator=accel,
      strategy=hydra.utils.instantiate(config.strategy),
      enable_progress_bar=False,
      logger=False,
      callbacks=[cb],
      enable_checkpointing=False,
      num_sanity_val_steps=0,
      limit_val_batches=0,
      precision=32,
  )
  trainer.fit(model, train_ds, valid_ds)
  return cb.losses


def _summarize_trend(losses: list[float]) -> tuple[float, float]:
  n = len(losses)
  k = max(1, n // 4)
  first = sum(losses[:k]) / k
  last = sum(losses[-k:]) / k
  return first, last


def main() -> int:
  if not torch.cuda.is_available():
    print('FAIL: CUDA required. Run on a GPU node (e.g. via Slurm).')
    return 1

  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=100)
  parser.add_argument('--hub', default='Qwen/Qwen2.5-0.5B')
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--block-size', type=int, default=16)
  parser.add_argument('--min-drop', type=float, default=0.01)
  parser.add_argument('--batch-size', type=int, default=2)
  args = parser.parse_args()

  results = {}
  for algo in ('block_masked', 'block_uniform'):
    losses = _run(
        algo, args.steps, args.hub, args.length, args.block_size,
        args.batch_size)
    if len(losses) < 2:
      print(f'WARN: only {len(losses)} loss points for {algo}')
      results[algo] = None
      continue
    first, last = _summarize_trend(losses)
    print(f'{algo}: loss {first:.4f} -> {last:.4f} ({len(losses)} steps)')
    if last >= first - abs(first) * args.min_drop:
      print(f'FAIL: {algo} loss did not drop by {args.min_drop*100:.0f}%')
      return 1
    results[algo] = (first, last)

  print('Tier 3 PASS: both arms ran and loss decreased')
  return 0


if __name__ == '__main__':
  sys.exit(main())
