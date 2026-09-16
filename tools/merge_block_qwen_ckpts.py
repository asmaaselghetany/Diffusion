#!/usr/bin/env python3
"""Linear-merge two Lightning block_qwen checkpoints (state_dict + ema).

Usage:
  python tools/merge_block_qwen_ckpts.py \\
    --chat outputs/.../1773298/checkpoints/last.ckpt \\
    --math outputs/.../1763301/checkpoints/last.ckpt \\
    --alpha 0.5 \\
    --out outputs/.../merged_a50/checkpoints/last.ckpt

``alpha`` = weight on the *chat* checkpoint (1-alpha on math).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _merge_tensor_dicts(a: dict, b: dict, alpha: float) -> dict:
  keys = set(a) | set(b)
  missing_a = sorted(keys - set(a))
  missing_b = sorted(keys - set(b))
  if missing_a or missing_b:
    raise KeyError(
        f'state_dict key mismatch: only_a={missing_a[:5]} only_b={missing_b[:5]}')
  out = {}
  for k in a:
    ta, tb = a[k], b[k]
    if not torch.is_floating_point(ta):
      out[k] = ta.clone() if torch.is_tensor(ta) else ta
      continue
    if ta.shape != tb.shape:
      raise ValueError(f'shape mismatch at {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}')
    out[k] = (alpha * ta.float() + (1.0 - alpha) * tb.float()).to(dtype=ta.dtype)
  return out


def merge_ckpts(chat_path: Path, math_path: Path, alpha: float, out_path: Path) -> None:
  if not (0.0 <= alpha <= 1.0):
    raise ValueError(f'alpha must be in [0,1], got {alpha}')
  chat = torch.load(chat_path, map_location='cpu', weights_only=False)
  math = torch.load(math_path, map_location='cpu', weights_only=False)
  if 'state_dict' not in chat or 'state_dict' not in math:
    raise KeyError('both checkpoints need Lightning state_dict')

  merged = dict(chat)  # keep chat hyper_parameters / optimizer scaffolding
  merged['state_dict'] = _merge_tensor_dicts(
      chat['state_dict'], math['state_dict'], alpha)

  if 'ema' in chat and 'ema' in math and chat['ema'] is not None and math['ema'] is not None:
    ea, eb = chat['ema'], math['ema']
    if isinstance(ea, dict) and isinstance(eb, dict):
      merged_ema = dict(ea)
      if 'shadows' in ea and 'shadows' in eb:
        merged_ema['shadows'] = _merge_tensor_dicts(ea['shadows'], eb['shadows'], alpha)
      if 'shadow_params' in ea and 'shadow_params' in eb:
        sa, sb = ea['shadow_params'], eb['shadow_params']
        if not (isinstance(sa, list) and isinstance(sb, list) and len(sa) == len(sb)):
          raise ValueError('ema.shadow_params list length mismatch')
        merged_list = []
        for i, (ta, tb) in enumerate(zip(sa, sb)):
          if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
            merged_list.append(ta)
            continue
          if ta.shape != tb.shape:
            raise ValueError(f'ema.shadow_params[{i}] shape mismatch')
          if torch.is_floating_point(ta):
            merged_list.append(
                (alpha * ta.float() + (1.0 - alpha) * tb.float()).to(dtype=ta.dtype))
          else:
            merged_list.append(ta.clone())
        merged_ema['shadow_params'] = merged_list
      # Keep decay/num_updates from chat.
      merged['ema'] = merged_ema

  out_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(merged, out_path)
  meta = {
      'chat_ckpt': str(chat_path),
      'math_ckpt': str(math_path),
      'alpha_chat': alpha,
      'note': 'linear merge of state_dict (+ ema if present); hparams from chat',
  }
  (out_path.parent.parent / 'MERGE.json').write_text(
      __import__('json').dumps(meta, indent=2) + '\n', encoding='utf-8')
  print(f'Wrote {out_path} (alpha_chat={alpha})')


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--chat', type=Path, required=True)
  ap.add_argument('--math', type=Path, required=True)
  ap.add_argument('--alpha', type=float, required=True, help='weight on chat ckpt')
  ap.add_argument('--out', type=Path, required=True)
  args = ap.parse_args()
  merge_ckpts(args.chat, args.math, args.alpha, args.out)


if __name__ == '__main__':
  main()
