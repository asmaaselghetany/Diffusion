#!/usr/bin/env python3
"""Build text8 train/valid disk caches for uni-d2 training.

Safe to run under a short file lock before launching parallel training jobs.
"""

from __future__ import annotations

import argparse
import os
import sys

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from discrete_diffusion.data.loaders import get_dataset, get_tokenizer
from discrete_diffusion.train import register_config_resolvers


def main() -> int:
  parser = argparse.ArgumentParser(description='Preprocess text8 dataset caches.')
  parser.add_argument(
      '--cache-dir',
      default=os.environ.get(
          'DATA_CACHE',
          '/fast/project/HFMI_SynergyUnit/asmaa.elsayed/.cache/discrete_diffusion/text8'))
  parser.add_argument('--length', type=int, default=128)
  parser.add_argument('--num-proc', type=int, default=4)
  args = parser.parse_args()

  config_dir = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

  register_config_resolvers()
  GlobalHydra.instance().clear()
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    config = compose(
        config_name='config',
        overrides=[
            'data=text8',
            'model=qwen_block',
            f'data.cache_dir={args.cache_dir}',
            f'model.length={args.length}',
            f'loader.num_workers={args.num_proc}',
        ])

  tokenizer = get_tokenizer(config)
  for mode, insert_eos in (
      ('train', config.data.insert_train_eos),
      ('test', config.data.insert_valid_eos),
  ):
    print(f'Preprocessing text8/{mode} -> {args.cache_dir}')
    get_dataset(
        config.data.train,
        tokenizer,
        wrap=config.data.wrap,
        mode=mode,
        cache_dir=config.data.cache_dir,
        insert_eos=insert_eos,
        insert_special_tokens=True,
        block_size=config.model.length,
        streaming=config.data.streaming,
        num_proc=config.loader.num_workers,
    )

  print('text8 cache ready:', args.cache_dir)
  return 0


if __name__ == '__main__':
  sys.exit(main())
