#!/usr/bin/env python3
"""Build an AR→block init checkpoint: Hub Fast_dLLM arch + Qwen Instruct weights.

Public ``finetune_alpaca.sh`` continues from ``Fast_dLLM_v2_*``. Our C2 convert
starts from Qwen2.5-1.5B-Instruct. Weight keys match 1:1 (verified), so we copy
Qwen tensors into a Hub Fast_dLLM template directory.

Usage:
  python tools/init_fastdllm_from_qwen.py \\
    --hub-template /e/scratch/.../Fast_dLLM_v2_1.5B \\
    --qwen /e/scratch/.../Qwen2.5-1.5B-Instruct \\
    --out /e/scratch/.../hub_fastdllm_c2/init_qwen15b
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


_SKIP_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


def _resolve_weight_file(root: Path) -> Path:
  single = root / "model.safetensors"
  if single.is_file():
    return single
  shards = sorted(root.glob("model-*.safetensors"))
  if shards:
    raise SystemExit(
        f"Sharded checkpoint at {root} not supported; flatten first.")
  bin_path = root / "pytorch_model.bin"
  if bin_path.is_file():
    return bin_path
  raise SystemExit(f"No model weights under {root}")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--hub-template", type=Path, required=True,
                  help="Local Fast_dLLM_v2_1.5B dir (config + modeling + tok).")
  ap.add_argument("--qwen", type=Path, required=True,
                  help="Local Qwen2.5-1.5B-Instruct dir.")
  ap.add_argument("--out", type=Path, required=True,
                  help="Output init directory.")
  ap.add_argument("--force", action="store_true",
                  help="Overwrite --out if it exists.")
  args = ap.parse_args()

  if not args.hub_template.is_dir():
    raise SystemExit(f"Missing hub template: {args.hub_template}")
  if not args.qwen.is_dir():
    raise SystemExit(f"Missing Qwen dir: {args.qwen}")
  if args.out.exists():
    if not args.force:
      raise SystemExit(f"Refusing to overwrite {args.out} (pass --force)")
    shutil.rmtree(args.out)
  args.out.mkdir(parents=True)

  # Copy Hub template files except weight blobs.
  for src in args.hub_template.iterdir():
    if src.name in _SKIP_NAMES or src.name.startswith("model-"):
      continue
    if src.name == ".cache":
      continue
    dst = args.out / src.name
    if src.is_dir():
      shutil.copytree(src, dst)
    else:
      shutil.copy2(src, dst)

  hub_w = _resolve_weight_file(args.hub_template)
  qwen_w = _resolve_weight_file(args.qwen)
  print(f"hub weights:  {hub_w}")
  print(f"qwen weights: {qwen_w}")

  if qwen_w.suffix == ".bin":
    import torch
    qwen_state = torch.load(qwen_w, map_location="cpu")
  else:
    qwen_state = load_file(str(qwen_w))

  if hub_w.suffix == ".bin":
    import torch
    hub_state = torch.load(hub_w, map_location="cpu")
  else:
    hub_state = load_file(str(hub_w))

  hub_keys = set(hub_state.keys())
  qwen_keys = set(qwen_state.keys())
  if hub_keys != qwen_keys:
    only_hub = sorted(hub_keys - qwen_keys)
    only_qwen = sorted(qwen_keys - hub_keys)
    raise SystemExit(
        f"Key mismatch: only_hub={only_hub[:10]} only_qwen={only_qwen[:10]}")

  # Shape check
  bad = []
  for k in sorted(hub_keys):
    if tuple(hub_state[k].shape) != tuple(qwen_state[k].shape):
      bad.append((k, tuple(hub_state[k].shape), tuple(qwen_state[k].shape)))
  if bad:
    raise SystemExit(f"Shape mismatch examples: {bad[:5]}")

  out_w = args.out / "model.safetensors"
  save_file(qwen_state, str(out_w))
  print(f"wrote {out_w} keys={len(qwen_state)}")

  meta = {
      "init": "qwen_into_fastdllm",
      "hub_template": str(args.hub_template),
      "qwen": str(args.qwen),
      "n_keys": len(qwen_state),
  }
  (args.out / "INIT_FROM_QWEN.json").write_text(
      json.dumps(meta, indent=2) + "\n", encoding="utf-8")
  print(f"init ready: {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
