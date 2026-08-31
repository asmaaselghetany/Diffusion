#!/usr/bin/env python3
"""Run decode_throughput with eval-time sampling overrides."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
  sys.path.insert(0, str(_REPO / 'src'))

from discrete_diffusion.evaluations.checkpoint_utils import load_block_trainer_checkpoint  # noqa: E402


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--metrics-path', required=True)
  parser.add_argument('--num-steps', type=int, default=32)
  parser.add_argument('--batch-size', type=int, default=1)
  parser.add_argument('--num-batches', type=int, default=4)
  parser.add_argument('--override', action='append', default=[])
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(False)
  model, config, tokenizer = load_block_trainer_checkpoint(
      args.checkpoint, device, hydra_overrides=args.override or None)
  sampler = model._create_sampler()
  if sampler is None:
    raise RuntimeError('no BlockSampler on checkpoint')

  num_steps = int(args.num_steps)
  batch_size = int(args.batch_size)
  prefix = tokenizer('Hello', add_special_tokens=False, return_tensors='pt')[
      'input_ids'].to(device)
  prefix_len = int(prefix.shape[1])

  def timed():
    if device.type == 'cuda':
      torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = sampler.generate(
        model, num_samples=batch_size, num_steps=num_steps,
        inject_bos=False, prefix_ids=prefix.expand(batch_size, -1))
    if device.type == 'cuda':
      torch.cuda.synchronize(device)
    return out, time.perf_counter() - t0

  for _ in range(1):
    timed()
  elapsed = 0.0
  tokens = 0
  for _ in range(int(args.num_batches)):
    samples, dt = timed()
    elapsed += dt
    cont = samples[:, prefix_len:]
    eos = tokenizer.eos_token_id
    for row in cont:
      if eos is not None:
        hits = (row == eos).nonzero(as_tuple=False)
        if hits.numel():
          row = row[: int(hits[0]) + 1]
      tokens += int((row != model.mask_id).sum().item())

  metrics = {
      'checkpoint_path': str(Path(args.checkpoint).resolve()),
      'sampling_overrides': args.override,
      'num_steps': num_steps,
      'batch_size': batch_size,
      'num_batches': args.num_batches,
      'tokens_generated': tokens,
      'elapsed_s': elapsed,
      'tok_s': tokens / max(elapsed, 1e-9),
  }
  out_path = Path(args.metrics_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
  print(json.dumps(metrics, indent=2))
  return 0


if __name__ == '__main__':
  sys.exit(main())
