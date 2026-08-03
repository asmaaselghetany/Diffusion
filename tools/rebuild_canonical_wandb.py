#!/usr/bin/env python3
"""Rebuild exactly 4 clean WandB runs from offline Lightning logs.

Uses wandb's protobuf DataStore (not regex heuristics) so all logged keys
are preserved: trainer/loss, trainer/lr, val/{nll,bpd,ppl,sample_entropy,...}.

Canonical run ids (name without _v3 suffix):
  ar2block_masked_v3, ar2block_uniform_v3, block_masked_v3, block_uniform_v3

X-axis is trainer/global_step (training step), not wandb's internal _step.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ENTITY = "asmaaselghetany-elghitany"
PROJECT = "block_qwen"

ARMS = {
    "ar2block_masked": Path("outputs/block_qwen/ar2block_masked_131655/hydra/wandb"),
    "ar2block_uniform": Path("outputs/block_qwen/ar2block_uniform_133161/hydra/wandb"),
    "block_masked": Path("outputs/block_qwen/block_masked_133150/hydra/wandb"),
    "block_uniform": Path("outputs/block_qwen/block_uniform_133151/hydra/wandb"),
}

# Keep charts focused; drop table/media nested keys.
KEEP_PREFIXES = (
    "trainer/",
    "train/",
    "val/",
    "lr-",
    "epoch",
)
SKIP_EXACT = {"trainer/global_step"}  # used as x-axis only


def _history_payload(rec: wandb_internal_pb2.Record) -> dict:
  """Flatten a HistoryRecord into {key: scalar}."""
  payload: dict = {}
  for item in rec.history.item:
    keys = list(item.nested_key) or ([item.key] if item.key else [])
    if not keys:
      continue
    # Nested media/table keys: samples@.../sha256 etc.
    if len(keys) > 1:
      continue
    key = keys[0]
    try:
      val = json.loads(item.value_json)
    except Exception:
      continue
    if isinstance(val, bool) or val is None:
      continue
    if isinstance(val, (int, float)):
      payload[key] = float(val) if not isinstance(val, bool) else val
    # skip strings / nested objects
  return payload


def parse_offline_root(root: Path) -> dict[int, dict]:
  """Merge all offline/online .wandb files → {global_step: metrics}."""
  files = sorted(root.glob("offline-run-*/*.wandb")) + sorted(root.glob("run-*/*.wandb"))
  by_step: dict[int, dict] = {}
  key_counts: Counter = Counter()
  for wb in files:
    if wb.stat().st_size < 64:
      print(f"  skip tiny {wb.parent.name}/{wb.name} ({wb.stat().st_size}B)")
      continue
    try:
      ds = DataStore()
      ds.open_for_scan(str(wb))
    except Exception as e:
      print(f"  skip unreadable {wb.parent.name}/{wb.name}: {e}")
      continue
    n_hist = 0
    while True:
      raw = ds.scan_data()
      if raw is None:
        break
      rec = wandb_internal_pb2.Record()
      try:
        rec.ParseFromString(raw)
      except Exception:
        continue
      if rec.WhichOneof("record_type") != "history":
        continue
      n_hist += 1
      payload = _history_payload(rec)
      gstep = payload.get("trainer/global_step")
      if gstep is None:
        continue
      step = int(gstep)
      if step < 0 or step > 20000:
        continue
      kept = {}
      for k, v in payload.items():
        if k in SKIP_EXACT:
          continue
        if not (k in ("epoch",) or k.startswith(KEEP_PREFIXES)):
          continue
        if not isinstance(v, (int, float)):
          continue
        # Sanity filters for corrupted console-bleed values
        if k.endswith("/loss") or k.endswith("/nll") or k.endswith("/bpd"):
          if not (0 < float(v) < 100):
            continue
        if k.endswith("/ppl") and not (1 < float(v) < 1e12):
          continue
        kept[k] = float(v)
      if not kept:
        continue
      by_step.setdefault(step, {}).update(kept)
      key_counts.update(kept.keys())
    print(f"  {wb.parent.name}: history_recs={n_hist}")
  print(f"  merged steps={len(by_step)} keys={dict(key_counts)}")
  return by_step


def wipe_canonical() -> None:
  api = wandb.Api(timeout=120)
  runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=200))
  print(f"wiping {len(runs)} existing runs in {ENTITY}/{PROJECT}")
  for r in runs:
    print(f"  delete name={r.name!r} id={r.id}")
    try:
      r.delete(delete_artifacts=True)
    except Exception as e:
      print(f"    warn: {e}")


def publish_arm(arm_id: str, root: Path) -> None:
  print(f"\n=== {arm_id} ===")
  if not root.is_dir():
    print(f"  SKIP missing {root}")
    return
  by_step = parse_offline_root(root)
  steps = sorted(by_step)
  if not steps:
    print(f"  SKIP empty {arm_id}")
    return

  # Fresh id suffix avoids DELETED-state collisions from prior deletes.
  wid = f"{arm_id}_v3"
  settings = wandb.Settings(init_timeout=300, console="off", silent=True)
  last_err: Exception | None = None
  for attempt in range(1, 4):
    try:
      run = wandb.init(
          project=PROJECT,
          entity=ENTITY,
          name=arm_id,
          id=wid,
          resume="allow",
          tags=["canonical", arm_id.split("_")[0], arm_id.split("_")[-1]],
          notes=(
              f"Canonical history for {arm_id} (protobuf rebuild). "
              "Live training resumes into this id."
          ),
          reinit=True,
          settings=settings,
      )
      break
    except Exception as e:
      last_err = e
      print(f"  wandb.init attempt {attempt} failed: {e}")
      try:
        wandb.finish(exit_code=1)
      except Exception:
        pass
  else:
    raise RuntimeError(f"wandb.init failed for {wid}") from last_err

  for i, step in enumerate(steps):
    payload = dict(by_step[step])
    payload["trainer/global_step"] = step
    wandb.log(payload, step=step)
    if (i + 1) % 100 == 0:
      print(f"  logged {i + 1}/{len(steps)}")
  wandb.finish()
  print(
      f"  published {len(steps)} steps "
      f"[{steps[0]}..{steps[-1]}] -> "
      f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{wid}"
  )


def main() -> None:
  os.environ.setdefault("WANDB_CONSOLE", "off")
  wipe_canonical()
  for arm_id, root in ARMS.items():
    publish_arm(arm_id, root)
  api = wandb.Api(timeout=120)
  runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
  print(f"\nFINAL {len(runs)} runs:")
  for r in sorted(runs, key=lambda x: x.name or ""):
    print(f"  name={r.name!r} id={r.id} state={r.state}")


if __name__ == "__main__":
  main()
