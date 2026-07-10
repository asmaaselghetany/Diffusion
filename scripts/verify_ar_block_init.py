#!/usr/bin/env python3
"""G5: AR→block init — load fraction + logit agreement at step 0."""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM

from discrete_diffusion.models.qwen.modeling import QwenBlockForCausalLM
from discrete_diffusion.training.init import compute_ar_block_init_metrics


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--hub', default='Qwen/Qwen2.5-0.5B')
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--block-size', type=int, default=16)
  parser.add_argument('--min-load', type=float, default=0.99)
  args = parser.parse_args()

  from omegaconf import OmegaConf
  config = OmegaConf.create({
      'model': {
          'hub_id': args.hub,
          'length': args.length,
          'forward_mode': 'block_diff',
          'attn_implementation': 'eager',
      },
      'block_size': args.block_size,
  })

  block_model = QwenBlockForCausalLM(config, vocab_size=0)
  hf_model = AutoModelForCausalLM.from_pretrained(
      args.hub, attn_implementation='eager')

  frac, missing, unexpected = block_model.compare_hf_keys(hf_model.state_dict())
  print(f'load_fraction={frac:.4f}')
  if frac < args.min_load:
    print('FAIL G5: load_fraction', missing[:5], unexpected[:5])
    return 1

  metrics = compute_ar_block_init_metrics(block_model, seed=0)
  print(f"logit_rho={metrics['logit_rho']:.4f}")
  print(f"top1_agreement={metrics['top1_agreement']:.4f}")
  print(f"max_logit_diff={metrics['max_logit_diff']:.4e}")

  if not all(map(torch.isfinite, [torch.tensor(v) for v in metrics.values()])):
    print('FAIL G5: non-finite init metrics')
    return 1

  print('G5 PASS')
  return 0


if __name__ == '__main__':
  sys.exit(main())
