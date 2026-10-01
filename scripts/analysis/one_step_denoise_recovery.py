#!/usr/bin/env python3
"""One-step Unif denoise recovery — oracle-N2C vs open-loop gap probe.

Teacher-forced dual-stream NLL can look healthy while ancestral generation is
soup. This script measures a cheaper intermediate:

  corrupt clean x0 @ mid-t → one forward (argmax x0) → accuracy on moved sites.

High ``acc_moved`` + soup GenPPL ⇒ multi-step / commit schedule issue.
Low ``acc_moved`` ⇒ train graph / weights cannot denoise even one step.

Compare ``--packing dual`` (train concat) vs ``single`` (``block_train_logits``).

Examples:
  .venv/bin/python scripts/analysis/one_step_denoise_recovery.py \\
    --ckpt outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt \\
    --packing dual --t 0.5 --n 32

  .venv/bin/python scripts/analysis/one_step_denoise_recovery.py \\
    --ckpt .../last.ckpt --packing single --t 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from discrete_diffusion.evaluations.checkpoint_utils import (
    load_block_trainer_checkpoint,
)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--ckpt', type=Path, required=True)
  p.add_argument(
      '--packing', choices=['dual', 'single'], default='dual',
      help='dual=concat(xt,x0) train graph; single=block_train_logits')
  p.add_argument('--t', type=float, default=0.5, help='Corruption time in (0,1)')
  p.add_argument('--n', type=int, default=32, help='Number of sequences')
  p.add_argument(
      '--seq-len', type=int, default=128,
      help='Truncate length (capped by model.length)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  p.add_argument('--out', type=Path, default=None)
  args = p.parse_args()

  torch.manual_seed(args.seed)
  device = torch.device(args.device)
  model, _cfg, _tok = load_block_trainer_checkpoint(args.ckpt, device)
  model.eval()
  bs = int(getattr(model, 'block_size', 32))
  length = min(int(model.num_tokens), int(args.seq_len))
  v = int(model.vocab_size)
  mask_id = int(model.mask_id)

  # Natural-language prompts (random vocab ids → chance ≈ 1/V; useless probe).
  tok = model.tokenizer
  texts = [
      '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
      '<|im_start|>user\nWhat is 12 + 35?<|im_end|>\n'
      '<|im_start|>assistant\n12 + 35 = 47. The answer is 47.<|im_end|>\n',
      '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
      '<|im_start|>user\nName three primary colors.<|im_end|>\n'
      '<|im_start|>assistant\nThe three primary colors are red, blue, and yellow.<|im_end|>\n',
      '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
      '<|im_start|>user\nWrite a short sentence about cats.<|im_end|>\n'
      '<|im_start|>assistant\nCats are curious animals that love to nap in sunny windows.<|im_end|>\n',
      '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
      '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n'
      '<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>\n',
  ]
  while len(texts) < args.n:
    texts.append(texts[len(texts) % 4])
  rows = []
  for i in range(args.n):
    ids = tok.encode(texts[i], add_special_tokens=False)
    if len(ids) < length:
      pad = int(getattr(tok, 'pad_token_id', None) or getattr(tok, 'eos_token_id', 0) or 0)
      ids = ids + [pad] * (length - len(ids))
    rows.append(ids[:length])
  x0 = torch.tensor(rows, device=device, dtype=torch.long)
  x0 = torch.where(x0 == mask_id, (x0 + 1) % v, x0)
  # Block FP expects t shaped [B, L] (shared within blocks at train time).
  # A bare [B] vector does not broadcast against [B, L] token draws.
  t = torch.full((args.n, length), float(args.t), device=device)

  with torch.no_grad():
    xt = model._corrupt(x0, t, block_size=bs)
    moved = xt != x0
    if args.packing == 'single':
      prev = bool(getattr(model, 'single_stream_train', False))
      model.single_stream_train = True
      try:
        logits = model._backbone_logits(
            xt, x0, block_size=bs, active_len=length)
      finally:
        model.single_stream_train = prev
    else:
      prev = bool(getattr(model, 'single_stream_train', False))
      model.single_stream_train = False
      try:
        logits = model._backbone_logits(
            xt, x0, block_size=bs, active_len=length)
      finally:
        model.single_stream_train = prev
    pred = logits.argmax(dim=-1)
    L = min(pred.shape[1], x0.shape[1])
    pred, x0_c, moved_c = pred[:, :L], x0[:, :L], moved[:, :L]
    acc_all = (pred == x0_c).float().mean().item()
    if moved_c.any():
      acc_moved = (pred[moved_c] == x0_c[moved_c]).float().mean().item()
    else:
      acc_moved = float('nan')
    move_rate = moved_c.float().mean().item()

  report = {
      'ckpt': str(args.ckpt.resolve()),
      'packing': args.packing,
      't': args.t,
      'n': args.n,
      'seq_len': length,
      'block_size': bs,
      'move_rate': move_rate,
      'acc_all': acc_all,
      'acc_moved': acc_moved,
      'note': (
          'acc_moved = argmax-x0 accuracy on Unif-corrupted sites (one forward, '
          'not full ancestral). dual=oracle N2C; single=decode packing.'
      ),
  }
  text = json.dumps(report, indent=2)
  print(text)
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + '\n')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
