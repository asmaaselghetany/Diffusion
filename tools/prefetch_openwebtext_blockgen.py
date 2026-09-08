#!/usr/bin/env python3
"""Prefetch BlockGen OWT (jdeschena/openwebtext) + Qwen wrapped .dat caches.

Login node only (compute nodes run HF_HUB_OFFLINE=1). Matches
third_party/blockgen/dataloader.py splits; tokenizer is Qwen for our backbone.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from discrete_diffusion.data.loaders import (  # noqa: E402
    get_dataset,
    get_tokenizer,
)


def _cache_dir() -> str:
  cache_dir = os.environ.get(
      "DATA_CACHE",
      os.path.join(
          os.environ.get("DISCRETE_DIFFUSION_SCRATCH_DIR", ""),
          "openwebtext-blockgen-qwen",
      ),
  )
  if not cache_dir:
    raise SystemExit("Set DATA_CACHE or DISCRETE_DIFFUSION_SCRATCH_DIR")
  os.makedirs(cache_dir, exist_ok=True)
  return cache_dir


def main() -> None:
  cache_dir = _cache_dir()
  num_proc = min(8, max(1, (os.cpu_count() or 4) // 2))
  # BlockGen small-block-dit default length; our structural block = 16.
  block_size = int(os.environ.get("BLOCKGEN_OWT_SEQ_LEN", "1024"))
  attention_block_size = int(os.environ.get("BLOCKGEN_OWT_BLOCK", "16"))
  tokenizer_name = os.environ.get(
      "BLOCK_QWEN_TOKENIZER", "Qwen/Qwen2.5-1.5B-Instruct")

  print(
      f"cache_dir={cache_dir} num_proc={num_proc} "
      f"length={block_size} block={attention_block_size}",
      flush=True)

  class _Cfg:
    data = type("D", (), {"tokenizer_name_or_path": tokenizer_name})()

  tokenizer = get_tokenizer(_Cfg())
  common = dict(
      tokenizer=tokenizer,
      wrap=True,
      cache_dir=cache_dir,
      block_size=block_size,
      insert_eos=True,
      insert_special_tokens=True,
      attention_block_size=attention_block_size,
      diffusion_block_size=attention_block_size,
      num_proc=num_proc,
      streaming=False,
  )
  train = get_dataset(
      "openwebtext-blockgen-train", mode="train", **common)
  print(f"Train .dat OK: {len(train)} chunks", flush=True)
  valid = get_dataset(
      "openwebtext-blockgen-valid", mode="validation", **common)
  print(f"Valid .dat OK: {len(valid)} chunks", flush=True)


if __name__ == "__main__":
  main()
