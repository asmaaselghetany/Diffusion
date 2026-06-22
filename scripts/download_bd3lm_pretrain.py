#!/usr/bin/env python3
"""Download BD3-LM OWT MDLM pretrain for offline BlockDiffusion finetuning.

Caches the Hugging Face weights under data_cache/hf_hub and writes a local
Lightning .ckpt for Jupiter compute nodes (no internet).

Usage:
  python scripts/download_bd3lm_pretrain.py
  python scripts/download_bd3lm_pretrain.py --cache-dir /path/to/data_cache
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from discrete_diffusion.training.pretrained import (  # noqa: E402
  DEFAULT_BD3LM_OWT_PRETRAIN_HF,
  default_pretrain_ckpt_path,
  save_pretrained_ckpt,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--cache-dir',
    default=os.environ.get('DATA_CACHE_DIR', os.path.join(REPO_ROOT, 'data_cache')),
    help='JEDi data_cache root (default: DATA_CACHE_DIR or ./data_cache)',
  )
  parser.add_argument(
    '--hf-repo',
    default=DEFAULT_BD3LM_OWT_PRETRAIN_HF,
    help='Hugging Face model id',
  )
  parser.add_argument(
    '--output',
    default='',
    help='Output .ckpt path (default: <cache-dir>/checkpoints/bd3lm_owt_block1024_pretrain.ckpt)',
  )
  args = parser.parse_args()

  cache_dir = os.path.abspath(args.cache_dir)
  output = args.output or default_pretrain_ckpt_path(cache_dir)

  os.environ.setdefault('HF_HOME', os.path.join(cache_dir, 'hf_home'))
  os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(cache_dir, 'hf_hub'))

  print(f'Downloading {args.hf_repo} ...')
  print(f'  HF cache: {os.environ["HUGGINGFACE_HUB_CACHE"]}')
  print(f'  Output:   {output}')

  save_pretrained_ckpt(args.hf_repo, output)
  size_gb = os.path.getsize(output) / (1024 ** 3)
  print(f'Done ({size_gb:.2f} GB). Use FROM_PRETRAINED=1 in train_core.sh / owt.sh')


if __name__ == '__main__':
  main()
