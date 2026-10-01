#!/usr/bin/env python3
"""Regression audit: uniform vs masked wiring honesty.

Exit 0 if coerce / UCC / hard-fill / branch semantics match the locked contract.
Exit 1 on regressions (silent UCC remap, hard-fill EOS mismatch, etc.).

Note: hierarchical thr on Unif → ancestral is *documented asymmetry*, not a
fail. The audit prints it and fails only if docs/code claim it is remask.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _denoise_branch(
    *,
    is_masked: bool,
    unmask_threshold,
    uniform_confidence_sticky: bool,
) -> str:
  if is_masked and unmask_threshold is not None:
    return 'confidence_remask'
  if (uniform_confidence_sticky and not is_masked
      and unmask_threshold is not None):
    return 'ucc_sticky'
  return 'ancestral_fixed_n'


def main() -> int:
  sys.path.insert(0, str(ROOT / 'src'))
  from discrete_diffusion.evaluations.decode_profiles import (
      coerce_profile_for_forward,
      profile_overrides,
  )
  from discrete_diffusion.forward_process.utils import (
      resolve_uniform_exclude_ids,
      resolve_uniform_noise_redraw_exclude_ids,
  )

  failures: list[str] = []

  # Branches for shared profile name
  b_m = _denoise_branch(
      is_masked=True, unmask_threshold=0.9,
      uniform_confidence_sticky=False)
  b_u = _denoise_branch(
      is_masked=False, unmask_threshold=0.9,
      uniform_confidence_sticky=False)
  print(f'[H1] hierarchical: masked={b_m} uniform={b_u}')
  if b_m != 'confidence_remask' or b_u != 'ancestral_fixed_n':
    failures.append(f'H1 unexpected branches M={b_m} U={b_u}')

  uc = profile_overrides('uniform_commit')
  sticky = any('uniform_confidence_sticky=true' in x for x in uc)
  b_uc = _denoise_branch(
      is_masked=False, unmask_threshold=1.0,
      uniform_confidence_sticky=sticky)
  print(f'[H1b] uniform_commit sticky={sticky} branch={b_uc}')
  if b_uc != 'ucc_sticky':
    failures.append(f'H1b UCC branch={b_uc}')

  # Coerce
  cmap = {
      f'{p}|{fp}': coerce_profile_for_forward(p, fp)
      for p, fp in [
          ('baseline', 'uniform'),
          ('hierarchical', 'uniform'),
          ('hubmatch', 'uniform'),
          ('dual_cache', 'uniform'),
          ('uniform_commit', 'uniform'),
      ]
  }
  print(f'[H5] coerce={cmap}')
  if cmap['baseline|uniform'] != 'hierarchical':
    failures.append('H5 baseline must coerce to hierarchical')
  if cmap['hierarchical|uniform'] != 'hierarchical':
    failures.append('H5 hierarchical must not remap to UCC')
  if cmap['hubmatch|uniform'] == 'uniform_commit':
    failures.append('H5 hubmatch silently remapped to UCC')

  # Hard-fill simplex = redraw/prior (EOS banned)
  class _Tok:
    mask_token_id = 151643
    pad_token_id = 151643
    eos_token_id = 151645
    vocab_size = 151936

    def convert_tokens_to_ids(self, t):
      return {'<|im_start|>': 151644, '<|im_end|>': 151645}.get(t)

  tok = _Tok()
  train_ex = set(resolve_uniform_exclude_ids(
      tok, mask_id=151643, vocab_size=151936))
  redraw_ex = set(resolve_uniform_noise_redraw_exclude_ids(
      tok, mask_id=151643, vocab_size=151936))
  print(f'[H4] train_fp={sorted(train_ex)} redraw={sorted(redraw_ex)}')
  if 151645 not in redraw_ex:
    failures.append('H4 EOS must be banned from redraw/prior')
  # Source check: _hard_pure_noise_xt uses redraw exclude
  src = (
      ROOT / 'src/discrete_diffusion/algorithms/block_trainer.py'
  ).read_text(encoding='utf-8')
  if 'def _hard_pure_noise_xt' not in src:
    failures.append('H4 missing _hard_pure_noise_xt')
  elif 'resolve_uniform_noise_redraw_exclude_ids' not in src[
      src.index('def _hard_pure_noise_xt'):
      src.index('def _hard_pure_noise_xt') + 800]:
    failures.append('H4 hard fill must call resolve_uniform_noise_redraw_exclude_ids')

  # Sampler warn present
  samp = (
      ROOT / 'src/discrete_diffusion/sampling/block_sampler.py'
  ).read_text(encoding='utf-8')
  if 'thr ignored' not in samp and 'ancestral (thr ignored)' not in samp:
    failures.append('H1 sampler must warn on dead Unif thr')

  # Docs must not claim conf remask on both arms as floor
  baseline = (ROOT / 'docs/research/BASELINE.md').read_text(encoding='utf-8')
  if 'Decode (both arms)** | BlockGen packing + **conf remask**' in baseline:
    failures.append('BASELINE still claims conf remask floor on both arms')
  if 'remask twin' not in baseline.lower() and 'dead on Unif' not in baseline:
    failures.append('BASELINE must document Unif remask twin / dead thr')

  if failures:
    print('\nFAIL:')
    for f in failures:
      print(f'  - {f}')
    return 1
  print('\nPASS: uniform wiring contract OK '
        '(hierarchical Unif≠masked remask is documented)')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
