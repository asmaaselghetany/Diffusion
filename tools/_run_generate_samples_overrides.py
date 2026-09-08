#!/usr/bin/env python3
"""Run generate_samples with eval-time sampling overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
  sys.path.insert(0, str(_REPO / 'src'))

from discrete_diffusion.evaluations.checkpoint_utils import (  # noqa: E402
    load_block_trainer_checkpoint,
)
from discrete_diffusion.evaluations.decode_profiles import (  # noqa: E402
    write_samples_meta,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--samples-path', required=True)
  parser.add_argument('--num-samples', type=int, default=32)
  parser.add_argument('--batch-size', type=int, default=1)
  parser.add_argument('--sample-mode', default='auto')
  parser.add_argument('--max-new-tokens', type=int, default=512)
  parser.add_argument('--override', action='append', default=[])
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(False)
  model, config, _tokenizer = load_block_trainer_checkpoint(
      args.checkpoint, device, hydra_overrides=args.override or None)

  num_steps = None
  for o in args.override:
    if o.startswith('num_steps='):
      num_steps = int(o.split('=', 1)[1])

  chunks = []
  with tqdm.tqdm(total=args.num_samples, desc='samples') as pbar:
    for i in range(0, args.num_samples, args.batch_size):
      bs = min(args.batch_size, args.num_samples - i)
      chunks.append(model.generate_samples(
          num_samples=bs,
          num_steps=num_steps,
          sample_mode=args.sample_mode,
          max_new_tokens=args.max_new_tokens,
      ).detach().cpu())
      pbar.update(bs)
  out = torch.cat(chunks, dim=0)
  path = Path(args.samples_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(out, path)
  write_samples_meta(path, {
      'checkpoint_path': str(Path(args.checkpoint).resolve()),
      'sample_mode': args.sample_mode,
      'decode_profile': 'keep',
      'max_new_tokens': args.max_new_tokens,
      'num_samples': int(out.shape[0]),
      'seq_len': int(out.shape[1]),
      'note': 'from tools/_run_generate_samples_overrides.py',
  })
  print(f'Saved {len(out)} samples to {path}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
