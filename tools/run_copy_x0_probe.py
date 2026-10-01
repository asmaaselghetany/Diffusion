#!/usr/bin/env python3
"""Copy-from-x0 / visibility diagnostic (diagnosis only — no redesign).

Corrects a false residual-weirdness claim: under block-diff attention,
``xt[i]`` does **not** attend ``x0[i]`` (same index / same block). The clean
half is *previous-block context*, not a visible answer key for the active
block. Floor low-t reconstruct is therefore a denoising failure (or
undertraining), not proof that \"x0 is visible but unused.\"

This probe reports:
  1. Mask facts (same-index / same-block xt→x0 forbidden; prev-block allowed).
  2. Position-id pattern HF actually uses on concat(xt, x0) length 2n.
  3. Empirical low-t denoise exact-on-holes on synthetic + optional real batch,
     stratified by block index (block0 has *no* x0 context).

Usage:
  python tools/run_copy_x0_probe.py --checkpoint path/to/best.ckpt --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from discrete_diffusion.models.block_mask import build_block_diff_bool_mask


def _load(checkpoint: str, device: str):
  from discrete_diffusion.evaluations.checkpoint_utils import (
      load_block_trainer_checkpoint,
  )
  from discrete_diffusion.train import register_config_resolvers
  from omegaconf import OmegaConf

  register_config_resolvers()
  model, config, tokenizer = load_block_trainer_checkpoint(
      checkpoint, torch.device(device))
  OmegaConf.resolve(config)
  return model, tokenizer, config


def mask_visibility_report(n: int, block_size: int) -> dict:
  m = build_block_diff_bool_mask(n, block_size)
  same_index = [bool(m[i, n + i].item()) for i in range(n)]
  same_block_any = []
  prev_block_ok = []
  for blk in range(n // block_size):
    xt_sl = slice(blk * block_size, (blk + 1) * block_size)
    x0_same = slice(n + blk * block_size, n + (blk + 1) * block_size)
    same_block_any.append(bool(m[xt_sl, x0_same].any()))
    if blk == 0:
      prev_block_ok.append(None)  # no previous
    else:
      x0_prev = slice(
          n + (blk - 1) * block_size, n + blk * block_size)
      prev_block_ok.append(bool(m[xt_sl, x0_prev].any()))
  return {
      'n': n,
      'block_size': block_size,
      'xt_to_same_index_x0_any_allowed': any(same_index),
      'xt_to_same_block_x0_any_allowed': any(same_block_any),
      'xt_block0_sees_any_x0': bool(m[0:block_size, n:].any()),
      'xt_later_block_sees_prev_x0': prev_block_ok[1] if len(prev_block_ok) > 1 else None,
      'interpretation': (
          'Same-index / same-block xt→x0 is forbidden by design '
          '(offset block-causal). Clean half ≠ visible answer key for '
          'active-block holes. Low-t floor ≠ failed copy-from-x0[i].'
      ),
  }


@torch.no_grad()
def exact_by_block(model, x0, t_val: float, block_size: int) -> dict:
  bsz, seq = x0.shape
  t = torch.full((bsz, seq), t_val, device=x0.device, dtype=torch.float32)
  xt = model._corrupt(x0, t, block_size=block_size)
  logits = model._backbone_logits(xt, x0, block_size=block_size)
  pred = logits.argmax(dim=-1)
  if model.forward_process_name == 'masked':
    holes = xt == model.mask_id
  else:
    holes = xt != x0
  out = {}
  n_blocks = seq // block_size
  for blk in range(n_blocks):
    sl = slice(blk * block_size, (blk + 1) * block_size)
    h = holes[:, sl]
    if not h.any():
      out[f'block_{blk}'] = {'exact': None, 'n_holes': 0}
      continue
    exact = float((pred[:, sl][h] == x0[:, sl][h]).float().mean().item())
    out[f'block_{blk}'] = {'exact': exact, 'n_holes': int(h.sum().item())}
  if holes.any():
    out['all'] = {
        'exact': float((pred[holes] == x0[holes]).float().mean().item()),
        'n_holes': int(holes.sum().item()),
    }
  else:
    out['all'] = {'exact': 1.0, 'n_holes': 0}
  return out


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--device', default='cuda')
  p.add_argument('--batch-size', type=int, default=2)
  p.add_argument('--seq-len', type=int, default=None)
  p.add_argument('--t', type=float, default=0.05, help='low-t probe')
  p.add_argument('--output', default=None)
  p.add_argument(
      '--real-batch', action='store_true',
      help='Also run on one train-loader batch if data cfg allows')
  args = p.parse_args()

  repo = Path(__file__).resolve().parents[1]
  if str(repo / 'src') not in sys.path:
    sys.path.insert(0, str(repo / 'src'))

  model, tokenizer, config = _load(args.checkpoint, args.device)
  seq = int(args.seq_len or config.model.length)
  bs = int(getattr(config, 'block_size', 32))
  # Model backbone may freeze n_tokens at train length.
  n_model = int(getattr(model.backbone, 'n_tokens', seq))
  seq = min(seq, n_model)

  vis = mask_visibility_report(seq, bs)

  # Position ids: concat(xt, x0) must share 0..n-1 (Fast-dLLM / BlockGen).
  from discrete_diffusion.models.qwen.modeling import shared_block_position_ids
  ids = shared_block_position_ids(seq, 'cpu', batch_size=1)
  pos_note = {
      'same_index_share_position_id': bool(
          (ids[0, :seq] == ids[0, seq:]).all().item()),
      'xt_ids': ids[0, : min(seq, 8)].tolist(),
      'x0_ids': ids[0, seq: seq + min(seq, 8)].tolist(),
      'note': (
          'QwenBlockForCausalLM.forward passes shared_block_position_ids: '
          'both halves use 0..n-1. Stock HF 0..2n-1 is not used.'
      ),
  }

  V = min(int(model.vocab_size), 1000)
  syn = torch.randint(1, V, (args.batch_size, seq), device=args.device)
  syn_rates = exact_by_block(model, syn, args.t, bs)

  real_rates = None
  if args.real_batch:
    try:
      from discrete_diffusion.data import get_dataloaders
      train_loader, _ = get_dataloaders(config, tokenizer)
      batch = next(iter(train_loader))
      x0 = batch['input_ids'][: args.batch_size, :seq].to(args.device)
      real_rates = exact_by_block(model, x0, args.t, bs)
    except Exception as e:  # noqa: BLE001 — probe must not crash the job
      real_rates = {'error': str(e)}

  report = {
      'checkpoint': args.checkpoint,
      'forward_process': model.forward_process_name,
      't': args.t,
      'seq': seq,
      'block_size': bs,
      'mask_visibility': vis,
      'position_ids': pos_note,
      'synthetic_low_t_exact_by_block': syn_rates,
      'real_batch_low_t_exact_by_block': real_rates,
      'verdict': (
          'If mask_visibility forbids same-index copy (expected), '
          'floor reconstruct is undertraining/denoising — not '
          '\"x0 visible but unused.\". Compare block_0 (no x0 context) '
          'vs later blocks (prev x0 context).'
      ),
  }
  print(json.dumps(report, indent=2))
  if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
