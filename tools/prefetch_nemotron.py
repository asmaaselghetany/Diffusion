#!/usr/bin/env python3
"""Download Nemotron SFT + build wrapped .dat caches (login node only)."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from discrete_diffusion.data.loaders import (  # noqa: E402
    _load_nemotron_sft,
    get_dataset,
    get_tokenizer,
)


def _cache_dir() -> str:
  cache_dir = os.environ.get(
      "DATA_CACHE",
      os.path.join(
          os.environ.get("DISCRETE_DIFFUSION_SCRATCH_DIR", ""),
          "block_qwen_sft_nemotron",
      ),
  )
  if not cache_dir:
    raise SystemExit("Set DATA_CACHE or DISCRETE_DIFFUSION_SCRATCH_DIR")
  os.makedirs(cache_dir, exist_ok=True)
  return cache_dir


def main() -> None:
  cache_dir = _cache_dir()
  num_proc = min(8, max(1, (os.cpu_count() or 4) // 2))
  block_size = int(os.environ.get("BLOCK_QWEN_SEQ_LEN", "2048"))
  tokenizer_name = os.environ.get(
      "BLOCK_QWEN_TOKENIZER", "Qwen/Qwen2.5-1.5B-Instruct")

  print(f"cache_dir={cache_dir} num_proc={num_proc} block_size={block_size}",
        flush=True)

  ds = _load_nemotron_sft(cache_dir=cache_dir, num_proc=num_proc, revision=None)
  print(f"Nemotron HF prefetch OK: {len(ds)} rows", flush=True)

  class _Cfg:
    data = type("D", (), {"tokenizer_name_or_path": tokenizer_name})()

  tokenizer = get_tokenizer(_Cfg())
  common = dict(
      tokenizer=tokenizer,
      wrap=True,
      cache_dir=cache_dir,
      block_size=block_size,
      insert_eos=False,
      insert_special_tokens=False,
      use_chat_template=os.environ.get("USE_CHAT_TEMPLATE", "1") != "0",
      num_proc=num_proc,
      streaming=False,
  )
  train = get_dataset("nemotron-sft-train", mode="train", **common)
  print(f"Train .dat OK: {len(train)} chunks", flush=True)
  valid = get_dataset("nemotron-sft-valid", mode="validation", **common)
  print(f"Valid .dat OK: {len(valid)} chunks", flush=True)


if __name__ == "__main__":
  main()
