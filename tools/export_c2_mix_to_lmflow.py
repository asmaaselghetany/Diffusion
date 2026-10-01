#!/usr/bin/env python3
"""Export our C2 Nemotron mix to LMFlow conversation JSON for Hub Fast-dLLM train.

Matches conversion_baseline caps (math≤1M, code≤500k, chat/safety/science full)
unless overridden via NEMOTRON_SFT_SPLITS / NEMOTRON_SFT_MAX_PER_SPLIT.

Writes one ``*.json`` per split under ``--out-dir`` (LMFlow loads all of them).

Usage:
  PYTHONPATH=src HF_HOME=... python tools/export_c2_mix_to_lmflow.py \\
    --out-dir /e/scratch/scifi/elsayed3/hub_fastdllm_c2/data/c2_mix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset

# Repo src on path when launched as tools/*.py
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

from discrete_diffusion.data.conversion_baseline import (  # noqa: E402
    nemotron_max_for_split,
    nemotron_resolved_caps,
    nemotron_split_list,
)
from discrete_diffusion.data.loaders import (  # noqa: E402
    _NEMOTRON_CONFIG,
    _NEMOTRON_HUB,
    _nemotron_to_messages,
)


def _row_to_instance(row: dict, idx: int, split: str) -> dict | None:
  """Map one Nemotron row → LMFlow conversation instance."""
  messages = _nemotron_to_messages(row)
  if not messages:
    return None
  system = ""
  body: list[dict[str, str]] = []
  for m in messages:
    role = (m.get("role") or "").lower()
    content = (m.get("content") or "").strip()
    if not content:
      continue
    if role == "system" and not system and not body:
      system = content
      continue
    if role not in {"user", "assistant", "system"}:
      continue
    body.append({"role": role, "content": content})
  if not any(m["role"] == "assistant" for m in body):
    return None
  if not any(m["role"] == "user" for m in body):
    return None
  inst: dict = {
      "conversation_id": f"{split}-{idx}",
      "messages": body,
  }
  if system:
    inst["system"] = system
  return inst


def _stream_write_split(path: Path, ds, split: str) -> tuple[Path, int]:
  """Stream LMFlow conversation JSON; return (final_path, kept_count)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".partial")
  kept = 0
  with tmp.open("w", encoding="utf-8") as f:
    f.write('{"type": "conversation", "instances": [\n')
    first = True
    for idx, row in enumerate(ds):
      inst = _row_to_instance(row, idx, split)
      if inst is None:
        continue
      if not first:
        f.write(",\n")
      json.dump(inst, f, ensure_ascii=False)
      first = False
      kept += 1
      if (idx + 1) % 50_000 == 0:
        print(f"  mapped {idx + 1}/{len(ds)} kept={kept}", flush=True)
    f.write("\n]}\n")
  final = path.parent / f"train_{split}_{kept}.json"
  tmp.replace(final)
  return final, kept


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      "--out-dir",
      type=Path,
      required=True,
      help="Directory for train_*.json shards (LMFlow --dataset_path).",
  )
  ap.add_argument(
      "--cache-dir",
      type=Path,
      default=None,
      help="HF datasets cache (default: HF_DATASETS_CACHE or HF_HOME/datasets).",
  )
  ap.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Shuffle seed before per-split cap (must match our loaders).",
  )
  ap.add_argument(
      "--dry-run",
      action="store_true",
      help="Print planned caps / row counts only; do not write JSON.",
  )
  args = ap.parse_args()

  cache_dir = args.cache_dir
  if cache_dir is None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets"))

  splits = nemotron_split_list()
  caps = dict(nemotron_resolved_caps())
  print("=== C2 → LMFlow export ===")
  print(f"  hub={_NEMOTRON_HUB} config={_NEMOTRON_CONFIG}")
  print(f"  splits={splits}")
  print(f"  caps={caps}")
  print(f"  cache={cache_dir}")
  print(f"  out={args.out_dir}")

  meta = {"splits": {}, "seed": args.seed, "hub": _NEMOTRON_HUB}
  total = 0
  for split in splits:
    print(f"--- load split={split} ---", flush=True)
    ds = load_dataset(
        _NEMOTRON_HUB,
        _NEMOTRON_CONFIG,
        split=split,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    n_full = len(ds)
    cap = nemotron_max_for_split(split)
    if cap is not None and n_full > cap:
      ds = ds.shuffle(seed=args.seed).select(range(cap))
      print(f"  subsample {n_full} -> {cap}", flush=True)
    else:
      ds = ds.shuffle(seed=args.seed)
      print(f"  keep full {n_full}", flush=True)

    if args.dry_run:
      meta["splits"][split] = {"n": len(ds), "full": n_full, "cap": cap}
      total += len(ds)
      continue

    # Stream to train_<split>_<n>.json
    out_stub = args.out_dir / f"train_{split}.json"
    print(f"  streaming write → {out_stub.parent}/train_{split}_*.json", flush=True)
    final_path, kept = _stream_write_split(out_stub, ds, split)
    meta["splits"][split] = {
        "n": kept,
        "full": n_full,
        "cap": cap,
        "file": final_path.name,
    }
    total += kept
    del ds
    print(f"  wrote {final_path} n={kept}", flush=True)

  meta["total"] = total
  if not args.dry_run:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Keep meta OUTSIDE the dataset_path glob (LMFlow loads every *.json).
    meta_path = args.out_dir.parent / f"{args.out_dir.name}_export_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {meta_path} total={total}", flush=True)
  else:
    print(json.dumps(meta, indent=2))
    print(f"dry-run total rows={total}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
