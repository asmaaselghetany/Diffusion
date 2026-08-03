#!/usr/bin/env python3
"""G2: block_diff forward shape and != causal."""

from __future__ import annotations

import argparse
import sys

import torch

from discrete_diffusion.models.qwen.modeling import QwenBlockForCausalLM


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--hub', default='Qwen/Qwen2.5-1.5B-Instruct')
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--block-size', type=int, default=8)
  args = parser.parse_args()

  from omegaconf import OmegaConf
  config = OmegaConf.create({
      'model': {
          'hub_id': args.hub, 'length': args.length,
          'forward_mode': 'block_diff', 'attn_implementation': 'eager',
      },
      'block_size': args.block_size,
  })

  model = QwenBlockForCausalLM(config, vocab_size=0)
  vocab = model.model.config.vocab_size
  n = args.length
  x0 = torch.randint(0, vocab, (1, n))
  concat = torch.cat([x0.clone(), x0], dim=-1)

  causal = model.causal_logits(x0)
  block = model.block_diff_logits(concat)

  if block.shape != (1, n, vocab):
    print('FAIL G2: bad shape', block.shape)
    return 1
  if torch.allclose(causal, block, atol=1e-5):
    print('FAIL G2: block == causal')
    return 1
  if not torch.isfinite(block).all():
    print('FAIL G2: non-finite logits')
    return 1
  print('G2 PASS')
  return 0


if __name__ == '__main__':
  sys.exit(main())
