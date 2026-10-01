#!/usr/bin/env python3
"""Layer 3–4 sanity for block_qwen arms (cheap, post-ckpt / no DepBench).

Checks:
  1. Self-consistency: reverse-process exact-match vs corruption level t
     should be roughly monotonic as t decreases (less corruption → easier).
  2. Denoising-to-self at low t: near-perfect reconstruction at t≈0+.

Usage (GPU):
  python tools/run_block_arm_sanity.py --checkpoint path/to/last.ckpt \\
      --device cuda --output outputs/block_qwen/sanity_uniform.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


@torch.no_grad()
def exact_after_corrupt_and_argmax(
    model, x0: torch.Tensor, t_val: float, *, block_size: int,
) -> float:
  """Corrupt at fixed t, one-step / teacher-forced read of model preference.

  For a cheap signal we score whether argmax(logits) at corrupted positions
  recovers x0 (masked: only mask sites; uniform: all sites). Full reverse
  diffusion is heavier; this isolates low-t posterior quality.
  """
  bsz, seq = x0.shape
  t = torch.full((bsz, seq), t_val, device=x0.device, dtype=torch.float32)
  xt = model._corrupt(x0, t, block_size=block_size)
  logits = model._backbone_logits(xt, x0, block_size=block_size)
  pred = logits.argmax(dim=-1)
  if model.forward_process_name == 'masked':
    holes = xt == model.mask_id
  else:
    holes = xt != x0
  if not holes.any():
    return 1.0
  return float((pred[holes] == x0[holes]).float().mean().item())


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--device', default='cuda')
  p.add_argument('--batch-size', type=int, default=4)
  p.add_argument('--seq-len', type=int, default=None)
  p.add_argument('--output', default=None)
  args = p.parse_args()

  repo = Path(__file__).resolve().parents[1]
  if str(repo / 'src') not in sys.path:
    sys.path.insert(0, str(repo / 'src'))

  from discrete_diffusion.evaluations.checkpoint_utils import (
      load_block_trainer_checkpoint,
  )
  from discrete_diffusion.train import register_config_resolvers
  from omegaconf import OmegaConf

  register_config_resolvers()
  device = torch.device(args.device)
  model, config, tokenizer = load_block_trainer_checkpoint(
      args.checkpoint, device)
  OmegaConf.resolve(config)
  seq = int(args.seq_len or config.model.length)
  bs = int(getattr(config, 'block_size', 32))
  # Synthetic clean sequences from vocab (not BOS-sensitive for this smoke).
  V = min(int(model.vocab_size), 1000)
  x0 = torch.randint(1, V, (args.batch_size, seq), device=args.device)

  # t grid: high t = more corruption under LogLinear.
  t_grid = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
  rates = []
  for t_val in t_grid:
    rates.append({
        't': t_val,
        'exact_on_corrupted': exact_after_corrupt_and_argmax(
            model, x0, t_val, block_size=bs),
    })

  # Monotonicity: exact should not systematically worsen as t decreases.
  # Compare adjacent pairs; allow small noise (±0.05).
  # At floor, the check is vacuous — do not report a bare true that looks
  # like a healthy posterior (uniform rates ~0–0.005 still "pass").
  FLOOR = 0.05
  all_at_floor = all(r['exact_on_corrupted'] < FLOOR for r in rates)
  mono_ok = True
  for i in range(len(rates) - 1):
    # lower t (earlier in list if sorted ascending) → higher exact
    if rates[i]['t'] < rates[i + 1]['t']:
      if rates[i]['exact_on_corrupted'] + 0.05 < rates[i + 1]['exact_on_corrupted']:
        # exact rose when corruption rose → suspicious
        mono_ok = False

  if all_at_floor:
    mono_status = 'inconclusive_floor'
    mono_ok_effective = False  # do not treat as a pass
  elif mono_ok:
    mono_status = 'ok'
    mono_ok_effective = True
  else:
    mono_status = 'fail'
    mono_ok_effective = False

  low_t = next(r for r in rates if r['t'] <= 0.05)
  low_t_ok = low_t['exact_on_corrupted'] >= 0.90

  report = {
      'checkpoint': args.checkpoint,
      'forward_process': model.forward_process_name,
      'rates': rates,
      # Legacy bool: False when inconclusive-at-floor (was misleadingly True).
      'monotonic_ok': mono_ok_effective,
      'monotonic_status': mono_status,
      'monotonic_raw_pairs_ok': mono_ok,
      'low_t_reconstruct_ok': low_t_ok,
      'low_t_exact': low_t['exact_on_corrupted'],
  }
  print(json.dumps(report, indent=2))
  if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
  return 0 if (mono_ok_effective and low_t_ok) else 3


if __name__ == '__main__':
  raise SystemExit(main())
