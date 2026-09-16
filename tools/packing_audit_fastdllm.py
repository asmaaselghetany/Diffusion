#!/usr/bin/env python3
"""Packing audit: our conversion baseline vs Hub Fast-dLLM (public v2).

Hub train packing is **not published** (LMFlow Dataset in v2/train_scripts).
This tool documents our invariants and prints the shared conversion
fingerprint so C0/C2/uniform jobs can verify they share one cache build.
"""

from __future__ import annotations

import json
import sys

from discrete_diffusion.data.conversion_baseline import (
    NEMOTRON_PREPROCESSING_VERSION,
    conversion_baseline_manifest,
)
from discrete_diffusion.data.processing import _group_block_aligned_sft


def _demo_invariants() -> dict:
  # Two short chats; block=4, seq=8. Expect pad-to-block then pack.
  examples = {
      'input_ids': [[1, 2, 3], [4, 5]],
      'attention_mask': [[1, 1, 1], [1, 1]],
      'labels': [[-100, 2, 3], [4, -100]],
  }
  out = _group_block_aligned_sft(
      examples,
      sequence_length=8,
      diffusion_block_size=4,
      mask_id=99,
  )
  rows = len(out['input_ids'])
  # Every packed row length == sequence_length
  ok_len = all(len(r) == 8 for r in out['input_ids'])
  # No diffusion block mixes two conversations: MASK pads to block boundary
  # before concat — checked by construction in unit tests.
  return {
      'demo_rows': rows,
      'demo_ok_row_len': ok_len,
      'demo_first_row': out['input_ids'][0] if rows else [],
  }


def main() -> int:
  manifest = conversion_baseline_manifest()
  demo = _demo_invariants()
  report = {
      'hub_fastdllm': {
          'public_packing': 'undisclosed',
          'public_train_entry': 'third_party/Fast-dLLM/v2/train_scripts/finetune.py',
          'notes': (
              'LMFlow Dataset wrapper; no block-aligned MASK pad published. '
              'Cannot claim bit-exact packing match.'
          ),
      },
      'our_conversion_packing': {
          'algorithm': manifest['packing'],
          'rules': [
              'Pad each conversation to diffusion_block_size with MASK',
              'attention_mask=0 / labels=-100 on that pad',
              'Concatenate; pad tail to sequence_length',
              'Drop rows with no supervised (non -100) labels',
          ],
          'intent': (
              'No diffusion block contains tokens from two conversations '
              '(block-attention / dual-stream safety).'
          ),
      },
      'shared_cache': {
          'preprocessing_version': NEMOTRON_PREPROCESSING_VERSION,
          'manifest': manifest,
          'fingerprint_uses': 'resolved (split, cap) not raw env string',
      },
      'demo': demo,
  }
  json.dump(report, sys.stdout, indent=2)
  sys.stdout.write('\n')
  if not demo['demo_ok_row_len']:
    return 1
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
