#!/usr/bin/env python3
"""Checkpoint-driven capability evaluation for continuous decoder runs.

Runs `scripts/eval_continuous_capabilities.py` on periodic checkpoints
(`...step=<N>.ckpt`) and logs scalar aggregates to W&B with `step=N`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


STEP_PATTERNS = (
    re.compile(r"step[=_-](\d+)"),
    re.compile(r"-(\d+)\.ckpt$"),
    re.compile(r"_(\d+)\.ckpt$"),
)


@dataclass(frozen=True)
class StepCheckpoint:
    step: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate continuous capabilities for each periodic checkpoint."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--preset", type=str, default="full", choices=["quick", "full"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampling_method", type=str, default="ddpm", choices=["ddpm", "ddim", "ancestral"])
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--qa_file", type=str, default="configs/eval/continuous_qa_prompts.sample.jsonl")
    parser.add_argument("--fixed_texts_file", type=str, default="configs/eval/continuous_fixed_texts.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--save_samples", action="store_true")

    parser.add_argument("--step_interval", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll_seconds", type=int, default=120)

    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="continuous-embedding-diffusion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    return parser.parse_args()


def _extract_step(path: Path) -> Optional[int]:
    name = path.name
    for pattern in STEP_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _discover_step_checkpoints(run_dir: Path) -> Dict[int, Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return {}

    by_step: Dict[int, Path] = {}
    for path in sorted(checkpoints_dir.glob("*.ckpt")):
        step = _extract_step(path)
        if step is None:
            continue
        current = by_step.get(step)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            by_step[step] = path
    return by_step


def _flatten_scalars(data: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_scalars(value, child_prefix))
        return out

    if isinstance(data, bool):
        out[prefix] = float(int(data))
        return out
    if isinstance(data, (int, float)):
        numeric = float(data)
        if math.isfinite(numeric):
            out[prefix] = numeric
        return out
    return out


def _collect_wandb_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    checkpoints = payload.get("checkpoints", [])
    if not checkpoints:
        return {}

    first = checkpoints[0]
    tasks = first.get("tasks", {})
    metrics: Dict[str, float] = {}

    for task_name, task_payload in tasks.items():
        for key, value in _flatten_scalars(task_payload).items():
            metrics[f"capability_eval/{task_name}/{key}"] = value

    return metrics


def _eval_output_dir(output_root: Path, step: int) -> Path:
    return output_root / f"step_{step:06d}"


def _build_eval_command(args: argparse.Namespace, checkpoint_path: Path, output_dir: Path) -> list[str]:
    checkpoint_pattern = f"checkpoints/{checkpoint_path.name}"
    cmd: list[str] = [
        sys.executable,
        "scripts/eval_continuous_capabilities.py",
        "--task",
        args.task,
        "--checkpoint_dir",
        str(checkpoint_path.parent.parent),
        "--checkpoint_pattern",
        checkpoint_pattern,
        "--preset",
        args.preset,
        "--device",
        args.device,
        "--sampling_method",
        args.sampling_method,
        "--qa_file",
        args.qa_file,
        "--fixed_texts_file",
        args.fixed_texts_file,
        "--output_dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--max_checkpoints",
        "1",
    ]

    if args.eta is not None:
        cmd.extend(["--eta", str(args.eta)])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.top_p is not None:
        cmd.extend(["--top_p", str(args.top_p)])
    if args.top_k is not None:
        cmd.extend(["--top_k", str(args.top_k)])
    if args.skip_ppl:
        cmd.append("--skip_ppl")
    if args.save_samples:
        cmd.append("--save_samples")

    return cmd


def _load_step_payload(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation output: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _wandb_init(args: argparse.Namespace, run_dir: Path):
    if args.wandb_mode == "disabled":
        return None

    import wandb

    run_name = args.wandb_run_name or f"{run_dir.name}_capability_eval"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group or run_dir.name,
        id=args.wandb_id,
        resume="allow" if args.wandb_id else None,
        mode=args.wandb_mode,
        config={
            "run_dir": str(run_dir),
            "task": args.task,
            "preset": args.preset,
            "step_interval": args.step_interval,
            "max_steps": args.max_steps,
        },
    )
    return run


def _evaluate_step(
    args: argparse.Namespace,
    step_ckpt: StepCheckpoint,
    output_root: Path,
    wandb_run,
) -> Dict[str, Any]:
    output_dir = _eval_output_dir(output_root, step_ckpt.step)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_eval_command(args, step_ckpt.path, output_dir)
    subprocess.run(cmd, check=True)

    payload = _load_step_payload(output_dir)
    record = {
        "step": step_ckpt.step,
        "checkpoint_path": str(step_ckpt.path),
        "output_dir": str(output_dir),
        "results_json": str(output_dir / "results.json"),
        "results_csv": str(output_dir / "results.csv"),
        "status": "ok",
    }

    if wandb_run is not None:
        metrics = _collect_wandb_metrics(payload)
        metrics["capability_eval/checkpoint_step"] = float(step_ckpt.step)
        wandb_run.log(metrics, step=step_ckpt.step)
        wandb_run.summary[f"capability_eval_step_{step_ckpt.step}_results_json"] = record["results_json"]
        wandb_run.summary["capability_eval_latest_step"] = step_ckpt.step

    return record


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else run_dir / "capability_eval_10k"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    expected_steps = list(range(args.step_interval, args.max_steps + 1, args.step_interval))
    done_steps = set()

    manifest_path = output_root / "checkpoint_eval_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        for record in previous.get("evaluations", []):
            if record.get("status") == "ok":
                done_steps.add(int(record["step"]))
    else:
        previous = {"evaluations": []}

    manifest: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "task": args.task,
        "preset": args.preset,
        "step_interval": args.step_interval,
        "max_steps": args.max_steps,
        "watch": bool(args.watch),
        "evaluations": list(previous.get("evaluations", [])),
    }

    wandb_run = _wandb_init(args, run_dir)

    try:
        while True:
            discovered = _discover_step_checkpoints(run_dir)
            eligible = {
                step: path
                for step, path in discovered.items()
                if step in expected_steps and step not in done_steps
            }

            for step in sorted(eligible):
                step_ckpt = StepCheckpoint(step=step, path=eligible[step])
                print(f"[eval] step={step_ckpt.step} checkpoint={step_ckpt.path}")
                try:
                    record = _evaluate_step(args, step_ckpt, output_root, wandb_run)
                    done_steps.add(step_ckpt.step)
                except Exception as exc:  # pylint: disable=broad-except
                    record = {
                        "step": step_ckpt.step,
                        "checkpoint_path": str(step_ckpt.path),
                        "output_dir": str(_eval_output_dir(output_root, step_ckpt.step)),
                        "status": "error",
                        "error": str(exc),
                    }
                    if wandb_run is not None:
                        wandb_run.log({"capability_eval/error": 1.0}, step=step_ckpt.step)
                manifest["evaluations"].append(record)
                _write_manifest(manifest_path, manifest)

            missing_steps = [step for step in expected_steps if step not in done_steps]
            if not missing_steps:
                print("[eval] completed all expected steps")
                break

            if not args.watch:
                print(f"[eval] missing checkpoints for steps: {missing_steps}")
                break

            print(f"[eval] waiting for checkpoints; next missing step={missing_steps[0]}")
            time.sleep(max(args.poll_seconds, 1))

    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"[eval] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
