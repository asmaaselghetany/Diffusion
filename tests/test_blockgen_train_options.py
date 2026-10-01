"""Fast-dLLM soft vs BlockGen hard train options (side-by-side)."""

from __future__ import annotations

import torch

from discrete_diffusion.models.block_mask import build_block_diff_bool_mask


def test_x0_causal_forbids_clean_lookahead():
  n, bs = 8, 4
  soft = build_block_diff_bool_mask(n, bs, 'cpu', x0_causal=False)
  hard = build_block_diff_bool_mask(n, bs, 'cpu', x0_causal=True)
  # Within first clean block: soft allows look-ahead; BlockGen forbids.
  q, kv = n + 1, n + 2
  assert soft[q, kv]
  assert not hard[q, kv]
  # Causal past still allowed under hard.
  assert hard[kv, q]


def test_xfer_bg_mix_presets_side_by_side():
  from pathlib import Path
  import yaml
  reg = yaml.safe_load(
      (Path(__file__).resolve().parents[1]
       / 'configs/levers/registry.yaml').read_text())
  soft = reg['presets']['xfer_bg_mix_32']['levers']
  bg = reg['presets']['xfer_bg_mix_32_blockgen']['levers']
  assert 'pure_noise_hard' not in soft
  assert 'x0_causal' not in soft
  assert 'pure_noise_hard' in bg
  assert 'x0_causal' in bg
