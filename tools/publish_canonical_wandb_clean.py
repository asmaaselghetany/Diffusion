#!/usr/bin/env python3
"""Wipe block_qwen and publish exactly 4 clean canonical WandB runs.

Each run gets:
  - scalar metrics from Lightning offline/online .wandb history
    (trainer/loss, trainer/lr, val/nll|bpd|ppl|collapse_*, …)
  - optional samples@global_stepN tables (off by default; in-train samples
    are excluded — paper eval samples are post-train only)

X-axis = trainer/global_step. Use WANDB_CANONICAL_SUFFIX for fresh run ids
when prior ids were deleted (WandB 409).
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ["WANDB_MODE"] = "online"

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ENTITY = "asmaaselghetany-elghitany"
PROJECT = "block_qwen"
RUN_SUFFIX = os.environ.get("WANDB_CANONICAL_SUFFIX", "v5")

ARMS = {
    "ar2block_masked": Path("outputs/block_qwen/ar2block_masked_131655/hydra/wandb"),
    "ar2block_uniform": Path("outputs/block_qwen/ar2block_uniform_133161/hydra/wandb"),
    "block_masked": Path("outputs/block_qwen/block_masked_133150/hydra/wandb"),
    "block_uniform": Path("outputs/block_qwen/block_uniform_133151/hydra/wandb"),
}

KEEP_EXACT = {"epoch"}
KEEP_PREFIXES = ("trainer/", "train/", "val/")
SKIP_KEYS = {
    "trainer/global_step",
    # In-train sample generation (turned off for paper loop) — do not publish.
    "val/sample_entropy",
    "val/samples",
    "val/samples_table",
    "val/sample_text",
    "val/sample_text_mid",
}
STEP_RE = re.compile(r"samples@global_step(\d+)_")


def _history_scalars(rec: wandb_internal_pb2.Record) -> dict:
  out: dict = {}
  for item in rec.history.item:
    keys = list(item.nested_key) or ([item.key] if item.key else [])
    if len(keys) != 1:
      continue
    key = keys[0]
    if key in SKIP_KEYS or key.startswith("samples@"):
      continue
    if not (key in KEEP_EXACT or key.startswith(KEEP_PREFIXES)):
      continue
    try:
      val = json.loads(item.value_json)
    except Exception:
      continue
    if isinstance(val, bool) or val is None:
      continue
    if isinstance(val, (int, float)):
      out[key] = float(val)
  return out


def parse_metrics(root: Path) -> dict[int, dict]:
  files = sorted(root.glob("offline-run-*/*.wandb")) + sorted(root.glob("run-*/*.wandb"))
  by_step: dict[int, dict] = {}
  key_counts: Counter = Counter()
  for wb in files:
    if wb.stat().st_size < 64:
      continue
    try:
      ds = DataStore()
      ds.open_for_scan(str(wb))
    except Exception as e:
      print(f"  skip {wb.parent.name}: {e}", flush=True)
      continue
    n = 0
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
      n += 1
      payload = _history_scalars(rec)
      gstep = payload.pop("trainer/global_step", None) if False else None
      # trainer/global_step was skipped above; re-read raw for step
    # Re-scan properly: need global_step from payload before skip
    print(f"  scanned {wb.parent.name}: history_recs≈{n}", flush=True)
  # Proper pass
  by_step = {}
  key_counts = Counter()
  for wb in files:
    if wb.stat().st_size < 64:
      continue
    try:
      ds = DataStore()
      ds.open_for_scan(str(wb))
    except Exception:
      continue
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
      # Extract step + scalars
      step = None
      kept: dict = {}
      for item in rec.history.item:
        keys = list(item.nested_key) or ([item.key] if item.key else [])
        if len(keys) != 1:
          continue
        key = keys[0]
        try:
          val = json.loads(item.value_json)
        except Exception:
          continue
        if key == "trainer/global_step" and isinstance(val, (int, float)):
          step = int(val)
          continue
        if key in SKIP_KEYS or key.startswith("samples@"):
          continue
        if not (key in KEEP_EXACT or key.startswith(KEEP_PREFIXES)):
          continue
        if isinstance(val, bool) or val is None or not isinstance(val, (int, float)):
          continue
        fval = float(val)
        if key.endswith(("/loss", "/nll", "/bpd")) and not (0 < fval < 100):
          continue
        if key.endswith("/ppl") and not (1 < fval < 1e12):
          continue
        kept[key] = fval
      if step is None or step < 0 or step > 20000 or not kept:
        continue
      by_step.setdefault(step, {}).update(kept)
      key_counts.update(kept.keys())
  print(f"  metrics steps={len(by_step)} keys={dict(key_counts)}", flush=True)
  return _fill_train_from_loss(by_step)


def _fill_train_from_loss(by_step: dict[int, dict]) -> dict[int, dict]:
  """Backfill train/nll|bpd|ppl from trainer/loss.

  Early block_qwen runs only logged trainer/loss; train/* was added later and
  is identical to loss (nll) / log2 / exp. Fill gaps so all arms have the same
  train metric keys across the full step range.
  """
  import math

  filled = 0
  for step, row in by_step.items():
    loss = row.get("trainer/loss")
    if loss is None:
      # Some rows only have train/nll — mirror into trainer/loss for plots.
      nll = row.get("train/nll")
      if nll is not None and "trainer/loss" not in row:
        row["trainer/loss"] = nll
        loss = nll
        filled += 1
    if loss is None:
      continue
    if "train/nll" not in row:
      row["train/nll"] = loss
      filled += 1
    if "train/bpd" not in row:
      row["train/bpd"] = loss / math.log(2)
      filled += 1
    if "train/ppl" not in row:
      try:
        row["train/ppl"] = math.exp(loss)
        filled += 1
      except OverflowError:
        pass
  print(f"  backfilled train/* from trainer/loss ({filled} fields)", flush=True)
  return by_step


def collect_samples(root: Path) -> dict[int, Path]:
  best: dict[int, Path] = {}
  for p in root.rglob("samples@global_step*.table.json"):
    m = STEP_RE.search(p.name)
    if not m:
      continue
    step = int(m.group(1))
    if step not in best or p.stat().st_mtime > best[step].stat().st_mtime:
      best[step] = p
  print(f"  sample tables={len(best)}", flush=True)
  return best


def wipe_project() -> None:
  api = wandb.Api(timeout=120)
  runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=200))
  print(f"wiping {len(runs)} runs in {ENTITY}/{PROJECT}", flush=True)
  for r in runs:
    print(f"  delete {r.name!r} ({r.id})", flush=True)
    try:
      r.delete(delete_artifacts=True)
    except Exception as e:
      print(f"    warn: {e}", flush=True)


def publish_arm(arm: str, root: Path) -> None:
  print(f"\n=== {arm} ===", flush=True)
  if not root.is_dir():
    print(f"  SKIP missing {root}", flush=True)
    return
  metrics = parse_metrics(root)
  skip_samples = os.environ.get("WANDB_SKIP_SAMPLES", "").lower() in {"1", "true", "yes"}
  samples = {} if skip_samples else collect_samples(root)
  steps = sorted(set(metrics) | set(samples))
  if not steps:
    print("  SKIP empty", flush=True)
    return

  wid = f"{arm}_{RUN_SUFFIX}"
  # Offline first: cluster→wandb.ai init often times out; sync afterward.
  run = wandb.init(
      project=PROJECT,
      entity=ENTITY,
      name=arm,
      id=wid,
      resume="allow",
      tags=["canonical", arm.split("_")[0], arm.split("_")[-1], RUN_SUFFIX],
      notes=(
          f"Clean canonical history for {arm}: scalars"
          + ("" if skip_samples else " + samples@global_stepN")
          + "."
      ),
      reinit=True,
      settings=wandb.Settings(mode="offline", console="off"),
  )
  print(
      f"  publishing {len(steps)} steps "
      f"(samples={'off' if skip_samples else len(samples)}) offline id={wid}",
      flush=True,
  )

  for i, step in enumerate(steps):
    payload: dict = {"trainer/global_step": step}
    payload.update(metrics.get(step, {}))
    if step in samples:
      raw = json.loads(samples[step].read_text())
      cols = raw.get("columns") or ["Generated Samples"]
      data = raw.get("data") or []
      if data:
        payload[f"samples@global_step{step}"] = wandb.Table(columns=cols, data=data)
    run.log(payload, step=step)
    if (i + 1) % 50 == 0:
      print(f"  logged {i + 1}/{len(steps)}", flush=True)

  run_dir = Path(run.dir).resolve().parent  # .../wandb/offline-run-.../files -> parent is run dir? 
  # run.dir is typically <run>/files; sync the run folder
  sync_path = Path(run.dir).resolve().parent
  run.finish()
  print(f"  offline done [{steps[0]}..{steps[-1]}] at {sync_path}", flush=True)

  # Push to cloud (may take a while; retries help on flaky links).
  import subprocess
  cmd = [
      "wandb", "sync",
      "--id", wid,
      "--append",
      "--skip-console",
      "-e", ENTITY,
      "-p", PROJECT,
      str(sync_path),
  ]
  print(f"  sync: {' '.join(cmd)}", flush=True)
  rc = subprocess.call(cmd)
  if rc != 0:
    print(f"  WARN: wandb sync exited {rc}", flush=True)
  else:
    print(
        f"  synced https://wandb.ai/{ENTITY}/{PROJECT}/runs/{wid}",
        flush=True,
    )


def main() -> int:
  # Isolate local wandb files on /tmp
  tmp = Path(f"/tmp/asmaa_wandb_clean_{RUN_SUFFIX}")
  tmp.mkdir(parents=True, exist_ok=True)
  os.environ["WANDB_DIR"] = str(tmp)
  os.environ["WANDB_CACHE_DIR"] = str(tmp / "cache")
  (tmp / "cache").mkdir(exist_ok=True)

  if os.environ.get("WANDB_SKIP_WIPE", "").lower() not in {"1", "true", "yes"}:
    wipe_project()
  else:
    print("skipping wipe (WANDB_SKIP_WIPE)", flush=True)
  for arm, root in ARMS.items():
    publish_arm(arm, root)

  api = wandb.Api(timeout=120)
  print("\nFINAL runs:", flush=True)
  for r in sorted(api.runs(f"{ENTITY}/{PROJECT}"), key=lambda x: x.name or ""):
    print(f"  name={r.name!r} id={r.id} state={r.state}", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
