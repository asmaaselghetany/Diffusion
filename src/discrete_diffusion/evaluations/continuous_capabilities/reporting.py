"""Reporting utilities for continuous capability evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple



def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out



def write_results_json(output_dir: str | Path, payload: Dict[str, Any]) -> Path:
    """Write the canonical JSON report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path



def _checkpoint_task_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for checkpoint in payload.get("checkpoints", []):
        base = {
            "run_name": checkpoint.get("run_name"),
            "checkpoint_path": checkpoint.get("checkpoint_path"),
        }

        metadata = checkpoint.get("metadata", {})
        for key, value in metadata.items():
            base[f"metadata.{key}"] = value

        tasks = checkpoint.get("tasks", {})
        for task_name, task_payload in tasks.items():
            row = dict(base)
            row["task"] = task_name
            row.update(_flatten_dict(task_payload))
            rows.append(row)
    return rows



def write_results_csv(output_dir: str | Path, payload: Dict[str, Any]) -> Path:
    """Write a flattened CSV report for easy spreadsheet analysis."""
    rows = _checkpoint_task_rows(payload)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.csv"

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run_name", "checkpoint_path", "task"])
        return path

    fieldnames: List[str] = []
    fieldname_set = set()
    for row in rows:
        for key in row.keys():
            if key not in fieldname_set:
                fieldname_set.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path



def write_sample_artifacts(output_dir: str | Path, payload: Dict[str, Any]) -> List[Path]:
    """Write per-task sample artifacts when available."""
    out_paths: List[Path] = []
    out_dir = Path(output_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint in payload.get("checkpoints", []):
        run_name = checkpoint.get("run_name", "unknown")
        safe_run_name = run_name.replace("/", "_")
        tasks = checkpoint.get("tasks", {})
        for task_name, task_payload in tasks.items():
            samples = task_payload.get("samples")
            if not samples:
                continue
            path = out_dir / f"{safe_run_name}__{task_name}.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                for item in samples:
                    handle.write(json.dumps(item, ensure_ascii=True) + "\n")
            out_paths.append(path)

    return out_paths


__all__ = [
    "write_results_json",
    "write_results_csv",
    "write_sample_artifacts",
]
