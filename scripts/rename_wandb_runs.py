#!/usr/bin/env python3
"""Clear, thesis-readable WandB display names for block_qwen runs.

Historical UNI-D2 recipes used short readable names (``mdlm_owt``,
``ar2block_masked``). Lever submits drifted to opaque tags like
``lever_C2_fdllm_full_…`` or ``lever_shift+complementary+…``.

Display-name scheme (id / resume key unchanged)::

    {Paradigm} · {corruption} · {recipe} · {jobid}

Examples::

    AR→block · masked · baseline · 1762534
    Joint · uniform · AR+causal · 1848844
    Native · masked · scratch · 1836796

Usage:
  python scripts/rename_wandb_runs.py              # rename all known axis runs
  JOBS=1762534,1849335 python scripts/rename_wandb_runs.py
  DRY_RUN=1 python scripts/rename_wandb_runs.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ENTITY = os.environ.get("WANDB_ENTITY", "aselghetany-nu")
PROJECT = os.environ.get("WANDB_TRAIN_PROJECT", "block_qwen")
BASE = Path(
    os.environ.get(
        "AXIS_OUTPUT_ROOT",
        "/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen",
    )
)

# job_id → (paradigm, corruption, recipe)  — recipe is the human role
CLEAR: Dict[str, Tuple[str, str, str]] = {
    "1836796": ("Native", "masked", "scratch"),
    "1762534": ("AR→block", "masked", "baseline"),
    "1763301": ("AR→block", "masked", "math-mix (all5)"),
    "1773298": ("AR→block", "masked", "chat-mix (all5)"),
    "1836800": ("AR→block", "masked", "mixed-SFT (all5)"),
    "1849335": ("AR→block", "uniform", "baseline"),
    "1855537": ("AR→block", "masked", "block-size mix"),
    "1849287": ("AR→block", "uniform", "block-size mix"),
    "1857323": ("AR→full", "n/a", "matched AR SFT"),
    "1836723": ("Joint", "masked", "AR+causal"),
    "1848844": ("Joint", "uniform", "AR+causal"),
    "1836725": ("Joint", "masked", "all5 + AR+causal"),
    "1856101": ("AR→block", "masked", "+shift"),
    "1856103": ("AR→block", "masked", "+complementary"),
    "1856105": ("AR→block", "masked", "+shift+comp"),
    "1857473": ("AR→block", "masked", "+intra-block anneal"),
    "1857278": ("AR→block", "uniform", "+shift"),
    "1857471": ("AR→block", "uniform", "+intra-block anneal"),
    "1857475": ("AR→block", "hybrid", "joint curriculum"),
    "1857519": ("AR→block", "uniform", "continue-FT from masked"),
    "1836811": ("AR→block", "hybrid", "p(uniform)=0.1"),
}


def clear_title(job: str, paradigm: str, corruption: str, recipe: str) -> str:
    return f"{paradigm} · {corruption} · {recipe} · {job}"


def title_for_job(job: str) -> Optional[str]:
    if job not in CLEAR:
        return None
    p, c, r = CLEAR[job]
    return clear_title(job, p, c, r)


def _train_run_id(run_dir: Path) -> Optional[str]:
    wandb_root = run_dir / "hydra" / "wandb"
    if not wandb_root.is_dir():
        return None
    offs = sorted(wandb_root.glob("offline-run-*"))
    if not offs:
        return None
    m = re.match(r"offline-run-\d+_\d+-(.+)$", offs[-1].name)
    return m.group(1) if m else None


def _run_dir(job: str) -> Optional[Path]:
    matches = sorted(BASE.glob(f"*_{job}"))
    return matches[0] if matches else None


def _load_key() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    candidates = [
        Path.home() / ".config/wandb/api_key",
    ]
    env_file = os.environ.get("WANDB_API_KEY_FILE", "").strip()
    if env_file:
        candidates.insert(0, Path(env_file).expanduser())
    for path in candidates:
        if path.is_file():
            os.environ["WANDB_API_KEY"] = path.read_text().strip()
            return
    raise SystemExit(
        "WANDB_API_KEY missing; export it or set WANDB_API_KEY_FILE / "
        "~/.config/wandb/api_key")


def rename_one(api, job: str, dry: bool) -> None:
    title = title_for_job(job)
    if not title:
        print(f"SKIP {job}: not in CLEAR map")
        return
    run_dir = _run_dir(job)
    if run_dir is None:
        print(f"SKIP {job}: no run dir")
        return
    rid = _train_run_id(run_dir)
    if not rid:
        print(f"SKIP {job}: no wandb id")
        return
    run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
    old = run.name
    if old == title:
        print(f"OK {job}: already '{title}'")
        return
    print(f"{job}: '{old}'  →  '{title}'")
    if dry:
        return
    run.name = title
    # Notes help the overview hover
    p, c, r = CLEAR[job]
    run.notes = (
        f"Paradigm: {p}\nCorruption: {c}\nRecipe: {r}\n"
        f"Job: {job}\nRun dir: {run_dir}\n"
        f"Stable id (resume): {rid}"
    )
    run.update()
    # Keep summary display fields in sync
    run.summary["display_name"] = title
    run.summary["paradigm"] = p
    run.summary["corruption"] = c
    run.summary["recipe"] = r
    run.summary.update()


def main() -> int:
    _load_key()
    import wandb

    dry = os.environ.get("DRY_RUN", "0") == "1"
    jobs = [
        j.strip()
        for j in os.environ.get("JOBS", ",".join(CLEAR.keys())).split(",")
        if j.strip()
    ]
    api = wandb.Api()
    for job in jobs:
        try:
            rename_one(api, job, dry=dry)
        except Exception as e:
            print(f"FAIL {job}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
