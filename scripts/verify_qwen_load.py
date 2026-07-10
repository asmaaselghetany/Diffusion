#!/usr/bin/env python3
"""G1: HF Qwen causal parity vs QwenBlockForCausalLM."""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM

from discrete_diffusion.models.qwen.modeling import QwenBlockForCausalLM


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--hub', default='Qwen/Qwen2.5-0.5B')
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--block-size', type=int, default=8)
  parser.add_argument('--max-logit-diff', type=float, default=1e-4)
  args = parser.parse_args()

  from omegaconf import OmegaConf
  config = OmegaConf.create({
      'model': {
          'hub_id': args.hub, 'length': args.length,
          'forward_mode': 'causal', 'attn_implementation': 'eager',
      },
      'block_size': args.block_size,
  })

  block_model = QwenBlockForCausalLM(config, vocab_size=0)
  hf_model = AutoModelForCausalLM.from_pretrained(
      args.hub, attn_implementation='eager')

  frac, missing, unexpected = block_model.compare_hf_keys(hf_model.state_dict())
  print(f'load_fraction={frac:.4f}')
  if frac < 0.99:
    print('FAIL G1: load_fraction < 0.99', missing[:5], unexpected[:5])
    return 1

  torch.manual_seed(0)
  ids = torch.randint(0, hf_model.config.vocab_size, (1, args.length))
  diff = (hf_model(ids, use_cache=False).logits - block_model.causal_logits(ids)).abs().max()
  print(f'max|logit_diff|={diff.item():.2e}')
  if diff.item() >= args.max_logit_diff:
    return 1
  print('G1 PASS')
  return 0


if __name__ == '__main__':
  sys.exit(main())
