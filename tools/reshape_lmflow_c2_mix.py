#!/usr/bin/env python3
"""Re-shard / cap LMFlow conversation JSON without breaking schema.

Naive line-splits of ``{"type","instances":[...]}`` drop the wrapper on
later parts → LMFlow ``"type" must be provided``. This tool streams
instances and writes **complete** LMFlow files.

Examples:
  # Cap Hub mix (avoids pyarrow 2GB offset overflow on full C2):
  python tools/reshape_lmflow_c2_mix.py \\
    --src .../c2_mix --dst .../c2_mix_hub \\
    --max-per-split chat=0,safety=0,math=40000,code=40000,science=40000

  # Valid size-bounded shards of one file:
  python tools/reshape_lmflow_c2_mix.py \\
    --src-file train_science_708920.json --dst-dir shards/ --max-bytes 400000000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _iter_instances(path: Path):
  """Yield instance dicts from an LMFlow conversation JSON (line-oriented).

  ``export_c2_mix_to_lmflow.py`` writes one instance per line after the header.
  """
  with path.open("r", encoding="utf-8") as f:
    header = f.readline()
    if '"type"' not in header or '"instances"' not in header:
      raise ValueError(
          f"{path}: missing LMFlow type/instances header: {header[:80]!r}")
    for line in f:
      s = line.strip().rstrip(",").strip()
      if not s.startswith("{"):
        continue
      if s.endswith("]}"):
        s = s[:-2].rstrip().rstrip(",")
      if not s.startswith("{"):
        continue
      yield json.loads(s)

def _write_lmflow(path: Path, instances: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".partial")
  with tmp.open("w", encoding="utf-8") as f:
    f.write('{"type": "conversation", "instances": [\n')
    for i, inst in enumerate(instances):
      if i:
        f.write(",\n")
      json.dump(inst, f, ensure_ascii=False)
    f.write("\n]}\n")
  tmp.replace(path)


def _split_name(path: Path) -> str:
  # train_math_1000000.json → math
  m = re.match(r"train_([a-zA-Z]+)_", path.name)
  return m.group(1) if m else path.stem


def _parse_caps(raw: str | None) -> dict[str, int]:
  """0 = keep all. Missing key = keep all."""
  out: dict[str, int] = {}
  if not raw:
    return out
  for part in raw.split(","):
    part = part.strip()
    if not part:
      continue
    k, v = part.split("=", 1)
    out[k.strip()] = int(v.strip())
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--src", type=Path, help="Directory of train_*.json")
  ap.add_argument("--dst", type=Path, help="Output directory")
  ap.add_argument("--src-file", type=Path, help="Single LMFlow JSON to shard")
  ap.add_argument("--dst-dir", type=Path, help="Shard output dir (with --src-file)")
  ap.add_argument(
      "--max-per-split",
      default="math=40000,code=40000,science=40000",
      help="Caps per split name; 0 or omit = all. chat/safety default all.",
  )
  ap.add_argument("--max-bytes", type=int, default=0, help="If >0, shard by approx size")
  args = ap.parse_args()

  caps = _parse_caps(args.max_per_split)

  if args.src_file:
    dst = args.dst_dir or args.dst
    if dst is None:
      raise SystemExit("--dst-dir required with --src-file")
    dst.mkdir(parents=True, exist_ok=True)
    stem = args.src_file.stem
    batch: list[dict] = []
    size = 0
    part = 0
    n = 0
    for inst in _iter_instances(args.src_file):
      blob = json.dumps(inst, ensure_ascii=False)
      if args.max_bytes and batch and size + len(blob) > args.max_bytes:
        out = dst / f"{stem}_part{part:02d}.json"
        _write_lmflow(out, batch)
        print(f"wrote {out.name} n={len(batch)}", flush=True)
        part += 1
        batch, size = [], 0
      batch.append(inst)
      size += len(blob) + 2
      n += 1
    if batch:
      out = dst / f"{stem}_part{part:02d}.json"
      _write_lmflow(out, batch)
      print(f"wrote {out.name} n={len(batch)}", flush=True)
    print(f"total instances {n}")
    return 0

  if not args.src or not args.dst:
    raise SystemExit("--src and --dst required (or --src-file/--dst-dir)")
  args.dst.mkdir(parents=True, exist_ok=True)
  files = sorted(args.src.glob("train_*.json"))
  if not files:
    raise SystemExit(f"no train_*.json under {args.src}")

  meta = {"src": str(args.src), "caps": caps, "files": []}
  for path in files:
    split = _split_name(path)
    cap = caps.get(split, 0)  # 0 = all
    kept: list[dict] = []
    for inst in _iter_instances(path):
      kept.append(inst)
      if cap and len(kept) >= cap:
        break
    out = args.dst / f"train_{split}_{len(kept)}.json"
    _write_lmflow(out, kept)
    print(f"{split}: {len(kept)} → {out.name} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    meta["files"].append({"split": split, "n": len(kept), "path": out.name})
  (args.dst / "META.txt").write_text(json.dumps(meta, indent=2) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
