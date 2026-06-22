#!/usr/bin/env python3
"""Prepare tokenized dataset caches for offline cluster training.

Downloads raw data and writes the wrapped ``.dat`` shards used by
``get_dataloaders``. Safe to re-run: existing caches are skipped.

Usage:
  python scripts/prep_data_cache.py --cache-dir ./data_cache \\
      --spec tiny_shakespeare:128 --spec openwebtext-split:1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (REPO_ROOT / "configs").as_posix()

sys.path.insert(0, str(REPO_ROOT / "src"))

from discrete_diffusion.data.loaders import get_dataset, get_tokenizer  # noqa: E402


def _validation_mode(data_config) -> str:
  if data_config.valid in ("text8", "lm1b", "ag_news"):
    return "test"
  return "validation"


def prep_one(config, num_workers: int) -> None:
  data = config.data
  block_size = config.model.length

  print("=" * 60)
  print(f"Dataset config : {data.train} / {data.valid}")
  print(f"Cache dir      : {data.cache_dir}")
  print(f"Block size     : {block_size}")
  print(f"Tokenizer      : {data.tokenizer_name_or_path}")
  print("=" * 60)

  tokenizer = get_tokenizer(config)
  print("Tokenizer ready.")

  train_ds = get_dataset(
    data.train,
    tokenizer,
    mode="train",
    wrap=data.wrap,
    insert_eos=data.insert_train_eos,
    insert_special_tokens=data.get("insert_train_special", True),
    cache_dir=data.cache_dir,
    block_size=block_size,
    streaming=data.streaming,
    num_proc=num_workers,
    revision=data.get("train_revision"),
    min_length=data.get("train_min_length", data.get("min_length", 0)),
    chunking=data.get("train_chunking", data.get("chunking", "none")),
  )
  print(f"Train cache ready ({len(train_ds)} examples).")

  valid_ds = get_dataset(
    data.valid,
    tokenizer,
    mode=_validation_mode(data),
    wrap=data.wrap,
    insert_eos=data.insert_valid_eos,
    insert_special_tokens=data.get("insert_valid_special", True),
    cache_dir=data.cache_dir,
    block_size=block_size,
    streaming=data.streaming,
    num_proc=num_workers,
    revision=data.get("valid_revision"),
    min_length=data.get("valid_min_length", data.get("min_length", 0)),
    chunking=data.get("valid_chunking", data.get("chunking", "none")),
  )
  print(f"Valid cache ready ({len(valid_ds)} examples).")


def parse_spec(spec: str) -> tuple[str, int]:
  if ":" not in spec:
    raise argparse.ArgumentTypeError(
      f"Expected DATA:SEQ_LEN, got {spec!r} (e.g. openwebtext-split:1024)"
    )
  data_name, seq_len_str = spec.rsplit(":", 1)
  data_name = data_name.strip()
  if not data_name:
    raise argparse.ArgumentTypeError(f"Missing data config name in {spec!r}")
  try:
    seq_len = int(seq_len_str)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
      f"Invalid seq len in {spec!r}: {seq_len_str!r}"
    ) from exc
  if seq_len < 1:
    raise argparse.ArgumentTypeError(f"SEQ_LEN must be >= 1, got {seq_len}")
  return data_name, seq_len


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Download and tokenize datasets into local .dat caches."
  )
  parser.add_argument(
    "--cache-dir",
    type=Path,
    required=True,
    help="Directory for HF caches and tokenized .dat shards.",
  )
  parser.add_argument(
    "--spec",
    type=parse_spec,
    action="append",
    help=(
      "Dataset Hydra config and block size, e.g. openwebtext-split:1024. "
      "Repeat for multiple datasets."
    ),
  )
  parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Parallel workers for tokenization (default: 8).",
  )
  parser.add_argument(
    "--pretrain-align",
    action="store_true",
    help=(
      "Match BD3-LM / FROM_PRETRAINED=1 data settings "
      "(no EOS or special tokens)."
    ),
  )
  args = parser.parse_args()

  specs = args.spec or [
    ("tiny_shakespeare", 128),
    ("openwebtext-split", 1024),
  ]

  cache_dir = args.cache_dir.resolve()
  cache_dir.mkdir(parents=True, exist_ok=True)

  with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
    for data_name, seq_len in specs:
      overrides = [
        f"data={data_name}",
        f"data.cache_dir={cache_dir.as_posix()}",
        f"model.length={seq_len}",
      ]
      if args.pretrain_align:
        overrides.extend([
          "data.insert_train_eos=False",
          "data.insert_valid_eos=False",
          "data.insert_train_special=False",
          "data.insert_valid_special=False",
        ])
      config = compose(config_name="config", overrides=overrides)
      prep_one(config, args.num_workers)

  print("=" * 60)
  print("All requested caches are ready.")
  print(f"Cache directory: {cache_dir}")
  print("=" * 60)


if __name__ == "__main__":
  main()
