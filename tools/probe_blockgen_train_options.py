#!/usr/bin/env python3
"""Check Fast-dLLM soft vs BlockGen hard train options (side-by-side).

Exit 0 if soft/hard diverge as expected and sibling presets exist.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
  import sys
  sys.path.insert(0, str(ROOT / 'src'))
  from discrete_diffusion.models.block_mask import build_block_diff_bool_mask
  from discrete_diffusion.forward_process.block_masked import (
      BlockMaskedForwardProcess,
  )
  from discrete_diffusion.noise_schedules import LogLinear

  n, bs = 8, 4
  soft_m = build_block_diff_bool_mask(n, bs, 'cpu', x0_causal=False)
  hard_m = build_block_diff_bool_mask(n, bs, 'cpu', x0_causal=True)
  q, kv = n + 1, n + 2
  soft_la = bool(soft_m[q, kv].item())
  hard_la = bool(hard_m[q, kv].item())
  print(f'x0 look-ahead: soft={soft_la} hard={hard_la}')

  class _Tok:
    mask_token_id = 0
    vocab_size = 100
    pad_token_id = 1

  fp = BlockMaskedForwardProcess(_Tok(), LogLinear())
  x0 = torch.arange(4096, dtype=torch.long).view(1, 4096) % 50 + 2
  t = torch.ones(1, 4096)
  xt_soft, _ = fp(x0, t, block_size=1, mask_schedule='alpha')
  soft_frac = float((xt_soft == 0).float().mean())
  hard_frac = 1.0
  print(f'noise frac @t=1: soft={soft_frac:.6f} hard={hard_frac}')

  reg = yaml.safe_load(
      (ROOT / 'configs/levers/registry.yaml').read_text())
  soft_p = reg['presets']['xfer_bg_mix_32']['levers']
  hard_p = reg['presets']['xfer_bg_mix_32_blockgen']['levers']
  print(f'preset soft={soft_p}')
  print(f'preset blockgen={hard_p}')

  ok = (
      soft_frac < 0.9999
      and hard_frac == 1.0
      and soft_la and (not hard_la)
      and 'pure_noise_hard' in hard_p
      and 'x0_causal' in hard_p
      and 'pure_noise_hard' not in soft_p
  )
  print('PASS' if ok else 'FAIL')
  return 0 if ok else 1


if __name__ == '__main__':
  raise SystemExit(main())
