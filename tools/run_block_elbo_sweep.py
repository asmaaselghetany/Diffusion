#!/usr/bin/env python3
"""Per-block-size validation ELBO sweep for a BlockTrainer checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _parse_int_list(text: str) -> list[int]:
  return [int(x.strip()) for x in text.split(',') if x.strip()]


def _load_model(checkpoint: Path, device: torch.device):
  import hydra.utils
  from discrete_diffusion.data import get_dataloaders, get_tokenizer
  from discrete_diffusion.train import register_config_resolvers

  register_config_resolvers()

  ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
  if 'hyper_parameters' not in ckpt or 'config' not in ckpt['hyper_parameters']:
    raise ValueError('Checkpoint missing hyper_parameters.config')
  config = ckpt['hyper_parameters']['config']
  if not OmegaConf.is_config(config):
    config = OmegaConf.create(config)

  # Checkpoints may retain unresolved Hydra interpolations (mul/div_up) and
  # multi-GPU batch math. Force single-node eval-compatible loader settings.
  OmegaConf.set_struct(config, False)
  num_gpus = max(1, torch.cuda.device_count())
  config.trainer.devices = num_gpus
  config.trainer.num_nodes = 1
  config.trainer.accumulate_grad_batches = 1
  config.loader.batch_size = 1
  config.loader.eval_batch_size = 1
  config.loader.eval_global_batch_size = num_gpus
  config.loader.global_batch_size = (
      int(config.loader.batch_size)
      * int(config.trainer.num_nodes)
      * num_gpus
      * int(config.trainer.accumulate_grad_batches))

  tokenizer = get_tokenizer(config)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  model = algo_cls.load_from_checkpoint(
      str(checkpoint), config=config, tokenizer=tokenizer, map_location=device)
  model.to(device)
  model.eval()

  _, valid_loader = get_dataloaders(config, tokenizer, skip_train=True)
  return model, valid_loader, config, tokenizer


@torch.no_grad()
def _eval_block_size(model, valid_loader, block_size: int,
                     max_batches: int, device: torch.device) -> dict:
  total_nll = 0.0
  total_tokens = 0.0
  batches = 0
  for batch in valid_loader:
    input_ids = batch['input_ids'].to(device)
    valid = batch['attention_mask'].to(device)
    nlls = model.nll(input_ids, valid, block_size=block_size)
    total_nll += float(nlls.sum().item())
    total_tokens += float(valid.sum().item())
    batches += 1
    if batches >= max_batches:
      break
  if total_tokens == 0:
    raise RuntimeError('No validation tokens seen')
  mean_nll = total_nll / total_tokens
  bpd = mean_nll / math.log(2)
  ppl = math.exp(mean_nll)
  return {
      'block_size': block_size,
      'mean_nll': mean_nll,
      'bpd': bpd,
      'ppl': ppl,
      'tokens': int(total_tokens),
      'batches': batches,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description='Block-size ELBO sweep')
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--block-sizes', default='1,4,16,32')
  parser.add_argument('--max-batches', type=int, default=50)
  parser.add_argument('--output', default=None)
  parser.add_argument('--device', default='cuda')
  args = parser.parse_args(argv)

  device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  checkpoint = Path(args.checkpoint).resolve()
  model, valid_loader, config, _ = _load_model(checkpoint, device)

  results = []
  for bs in _parse_int_list(args.block_sizes):
    if model.num_tokens % bs != 0:
      print(f'Skip block_size={bs}: does not divide seq_len={model.num_tokens}')
      continue
    row = _eval_block_size(model, valid_loader, bs, args.max_batches, device)
    # BlockGen-aligned: size-1 eval is CE under pure noise (not continuous ELBO).
    row['meter'] = 'ce_pure_noise' if bs == 1 else 'elbo'
    results.append(row)
    print(
        f'bs={row["block_size"]:2d}  [{row["meter"]}]  '
        f'nll={row["mean_nll"]:.4f}  '
        f'bpd={row["bpd"]:.4f}  ppl={row["ppl"]:.2f}  tokens={row["tokens"]}')

  payload = {
      'checkpoint': str(checkpoint),
      'forward_process': getattr(config.algo, 'forward_process_name', None),
      'seq_len': int(model.num_tokens),
      'default_block_size': int(model.block_size),
      'size1_eval': 'ce_pure_noise (BlockGen-aligned)',
      'results': results,
  }
  out = args.output or str(
      checkpoint.parent.parent / 'eval' / 'block_elbo_sweep.json')
  out_path = Path(out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2)
  print(f'Wrote {out_path}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
