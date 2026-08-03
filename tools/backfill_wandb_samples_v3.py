#!/usr/bin/env python3
"""Backfill val/samples tables onto canonical *_v3 WandB runs.

Reads Lightning-logged ``samples@global_step*.table.json`` from offline run
dirs and re-logs them under the stable key ``val/samples`` so the Samples
panel is visible on the rebuilt runs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ["WANDB_MODE"] = "offline"

import wandb

ENTITY = "asmaaselghetany-elghitany"
PROJECT = "block_qwen"

ARMS = {
    "ar2block_masked": Path("outputs/block_qwen/ar2block_masked_131655/hydra/wandb"),
    "ar2block_uniform": Path("outputs/block_qwen/ar2block_uniform_133161/hydra/wandb"),
    "block_masked": Path("outputs/block_qwen/block_masked_133150/hydra/wandb"),
    "block_uniform": Path("outputs/block_qwen/block_uniform_133151/hydra/wandb"),
}

STEP_RE = re.compile(r"samples@global_step(\d+)_")


def collect_tables(root: Path) -> dict[int, Path]:
  best: dict[int, Path] = {}
  for p in root.rglob("samples@global_step*.table.json"):
    m = STEP_RE.search(p.name)
    if not m:
      continue
    step = int(m.group(1))
    if step not in best or p.stat().st_mtime > best[step].stat().st_mtime:
      best[step] = p
  return best


def publish_arm(arm_id: str, root: Path) -> Path | None:
  tables = collect_tables(root)
  steps = sorted(tables)
  print(f"\n=== {arm_id}: {len(steps)} sample tables ===", flush=True)
  if not steps:
    return None

  wid = f"{arm_id}_v3"
  wandb.init(
      project=PROJECT,
      entity=ENTITY,
      name=arm_id,
      id=wid,
      resume="allow",
      reinit=True,
      settings=wandb.Settings(console="off", silent=True),
  )
  for i, step in enumerate(steps):
    raw = json.loads(tables[step].read_text())
    cols = raw.get("columns") or ["Generated Samples"]
    data = raw.get("data") or []
    if not data:
      continue
    wandb.log(
        {"val/samples": wandb.Table(columns=cols, data=data),
         "trainer/global_step": step},
        step=step,
    )
    if (i + 1) % 50 == 0:
      print(f"  logged {i + 1}/{len(steps)}", flush=True)
  wandb.finish()
  latest = Path("wandb/latest-run").resolve()
  print(f"  offline -> {latest}", flush=True)
  return latest


def main() -> int:
  offline_dirs: list[tuple[str, Path]] = []
  for arm_id, root in ARMS.items():
    out = publish_arm(arm_id, root)
    if out is not None:
      offline_dirs.append((f"{arm_id}_v3", out))

  print("\n=== SYNC ===", flush=True)
  os.environ["WANDB_MODE"] = "online"
  for wid, offline_dir in offline_dirs:
    print(f"syncing samples into {wid} from {offline_dir}", flush=True)
    proc = subprocess.run(
        ["wandb", "sync", "--id", wid, str(offline_dir)],
        check=False,
    )
    print(f"  sync rc={proc.returncode}", flush=True)
  print("DONE", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
