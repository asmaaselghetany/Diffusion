#!/usr/bin/env python3
"""Reorganize continuous embedding outputs into a canonical layout.

The script is non-destructive by design:
- Existing run/eval directories are moved into a structured tree.
- Backward-compatible symlinks are created at original locations.
- A manifest is generated for auditability.

Default mode is dry-run. Use --apply to perform changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


TIMESTAMP_RE = re.compile(r".*_\d{8}_\d{6}$")


@dataclass
class MovePlan:
    source: Path
    destination: Path
    category: str
    family: str
    stage: str
    action: str  # move_and_symlink | merge_capability_eval | skip
    note: Optional[str] = None


def _is_structured_dir(name: str) -> bool:
    return name in {"logs", "runs", "eval", "manifests"}


def _classify_run_dir(run_dir: Path, root: Path) -> MovePlan:
    name = run_dir.name

    if run_dir.is_symlink():
        return MovePlan(
            source=run_dir,
            destination=run_dir,
            category="system",
            family="symlink",
            stage="meta",
            action="skip",
            note="already an alias symlink",
        )

    if name == "capability_eval":
        return MovePlan(
            source=run_dir,
            destination=root / "eval" / "capability",
            category="evaluation",
            family="capability",
            stage="eval",
            action="merge_capability_eval",
        )

    if name == "logs":
        return MovePlan(
            source=run_dir,
            destination=run_dir,
            category="system",
            family="logs",
            stage="meta",
            action="skip",
            note="kept in place",
        )

    if _is_structured_dir(name):
        return MovePlan(
            source=run_dir,
            destination=run_dir,
            category="system",
            family=name,
            stage="meta",
            action="skip",
            note="already in canonical layout",
        )

    if name.startswith("decoder_robust_"):
        family = name.split("_from", 1)[0]
        stage = "stage2"
        destination = root / "runs" / stage / family / name
        return MovePlan(
            source=run_dir,
            destination=destination,
            category="training",
            family=family,
            stage=stage,
            action="move_and_symlink",
        )

    if any(
        name == prefix or name.startswith(f"{prefix}_")
        for prefix in ("quick_test", "decoder_only_baseline", "denoiser_x0", "denoiser_epsilon", "denoiser_v")
    ):
        family = (
            "denoiser_x0" if name.startswith("denoiser_x0")
            else "denoiser_epsilon" if name.startswith("denoiser_epsilon")
            else "denoiser_v" if name.startswith("denoiser_v")
            else "decoder_only_baseline" if name.startswith("decoder_only_baseline")
            else "quick_test"
        )
        has_timestamp = bool(TIMESTAMP_RE.match(name))
        stage = "stage1"
        if has_timestamp:
            destination = root / "runs" / stage / family / name
        else:
            destination = root / "runs" / stage / "legacy" / name
        return MovePlan(
            source=run_dir,
            destination=destination,
            category="training",
            family=family,
            stage=stage,
            action="move_and_symlink",
        )

    return MovePlan(
        source=run_dir,
        destination=root / "runs" / "unclassified" / name,
        category="unknown",
        family="unclassified",
        stage="unknown",
        action="move_and_symlink",
        note="unrecognized naming pattern",
    )


def _safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            raise FileExistsError(f"Cannot create symlink; path exists and is not symlink: {link_path}")
    rel_target = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel_target)


def _move_and_link(plan: MovePlan) -> None:
    plan.destination.parent.mkdir(parents=True, exist_ok=True)

    if plan.destination.exists():
        raise FileExistsError(f"Destination already exists: {plan.destination}")

    original_path = plan.source
    plan.source.rename(plan.destination)
    _safe_symlink(plan.destination, original_path)


def _merge_capability_eval(plan: MovePlan) -> None:
    source = plan.source
    destination = plan.destination
    destination.mkdir(parents=True, exist_ok=True)

    for child in sorted(source.iterdir()):
        target = destination / child.name
        if target.exists():
            raise FileExistsError(f"Cannot merge capability eval; target exists: {target}")
        child.rename(target)

    if source.exists() and not any(source.iterdir()):
        source.rmdir()
    _safe_symlink(destination, source)


def _iter_top_level_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def _render_plan(plans: List[MovePlan]) -> str:
    lines = []
    for plan in plans:
        line = f"[{plan.action}] {plan.source} -> {plan.destination} ({plan.stage}/{plan.family})"
        if plan.note:
            line += f" | {plan.note}"
        lines.append(line)
    return "\n".join(lines)


def _write_manifest(root: Path, plans: List[MovePlan], applied: bool) -> Path:
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": datetime.now().isoformat(),
        "root": str(root),
        "applied": applied,
        "plans": [
            {
                **asdict(plan),
                "source": str(plan.source),
                "destination": str(plan.destination),
            }
            for plan in plans
        ],
    }

    json_path = manifests_dir / f"continuous_outputs_manifest_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = manifests_dir / f"continuous_outputs_manifest_{stamp}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "destination", "category", "family", "stage", "action", "note"],
        )
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "source": str(plan.source),
                    "destination": str(plan.destination),
                    "category": plan.category,
                    "family": plan.family,
                    "stage": plan.stage,
                    "action": plan.action,
                    "note": plan.note or "",
                }
            )

    latest_json = manifests_dir / "continuous_outputs_manifest.json"
    latest_csv = manifests_dir / "continuous_outputs_manifest.csv"
    shutil.copy2(json_path, latest_json)
    shutil.copy2(csv_path, latest_csv)

    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize continuous embedding output directories")
    parser.add_argument(
        "--root",
        type=str,
        default="outputs/continuous_embedding",
        help="Root output directory to organize",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform filesystem changes (default is dry-run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    top_level_dirs = _iter_top_level_dirs(root)
    plans = [_classify_run_dir(d, root) for d in top_level_dirs]

    print("Planned actions:")
    print(_render_plan(plans))

    if args.apply:
        for plan in plans:
            if plan.action == "skip":
                continue
            if plan.action == "move_and_symlink":
                _move_and_link(plan)
            elif plan.action == "merge_capability_eval":
                _merge_capability_eval(plan)
            else:
                raise ValueError(f"Unknown action: {plan.action}")
        print("\nApplied filesystem changes.")
    else:
        print("\nDry-run only. Re-run with --apply to execute.")

    manifest_path = _write_manifest(root, plans, applied=args.apply)
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
