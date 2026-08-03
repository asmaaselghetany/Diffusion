#!/usr/bin/env python3
"""Offline rebuild of canonical WandB runs (_v3), then online sync."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Must set before importing wandb in callers; we set again for clarity.
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

import wandb

from tools.rebuild_canonical_wandb import ARMS, ENTITY, PROJECT, parse_offline_root


def main() -> int:
  os.environ["WANDB_MODE"] = "offline"
  out_roots: list[tuple[str, Path]] = []

  for arm_id, root in ARMS.items():
    print(f"\n=== OFFLINE {arm_id} ===", flush=True)
    by_step = parse_offline_root(root)
    steps = sorted(by_step)
    if not steps:
      print("SKIP empty", flush=True)
      continue
    wid = f"{arm_id}_v3"
    wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=arm_id,
        id=wid,
        resume="allow",
        tags=["canonical", arm_id.split("_")[0], arm_id.split("_")[-1]],
        notes=f"Canonical protobuf rebuild for {arm_id}. Live training resumes here.",
        reinit=True,
        settings=wandb.Settings(console="off", silent=True),
    )
    for i, step in enumerate(steps):
      payload = dict(by_step[step])
      payload["trainer/global_step"] = step
      wandb.log(payload, step=step)
      if (i + 1) % 100 == 0:
        print(f"  logged {i + 1}/{len(steps)}", flush=True)
    wandb.finish()
    latest = Path("wandb/latest-run").resolve()
    print(
        f"  offline dir {latest} steps={len(steps)} [{steps[0]}..{steps[-1]}]",
        flush=True,
    )
    out_roots.append((wid, latest))

  print("\n=== SYNC ===", flush=True)
  os.environ["WANDB_MODE"] = "online"
  for wid, offline_dir in out_roots:
    print(f"syncing {wid} from {offline_dir}", flush=True)
    proc = subprocess.run(
        ["wandb", "sync", "--id", wid, str(offline_dir)],
        check=False,
    )
    print(f"  sync rc={proc.returncode}", flush=True)
  print("DONE", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
