#!/usr/bin/env python3
"""Upload real on-disk eval artifacts + metrics to WandB for every axis cell.

For each presentation-axis run dir under outputs/block_qwen/*_<jid>:
  - resume/create companion run ``eval_<jid>`` in project ``thesis_axes``
  - log GenPPL / pair / hygiene / ELBO / lm_eval SUMMARY scores to summary
  - upload samples.txt + metric JSONs as a ``eval-bundle`` artifact
  - register ``last.ckpt`` as a model artifact *reference* (no 24G re-upload;
    set UPLOAD_CKPT=1 to stream the file)

Also patches the synced training run summary on ``block_qwen`` with eval
pointers when the train run id can be inferred from the offline wandb dir.

Usage:
  WANDB_API_KEY=... python scripts/upload_axis_eval_artifacts.py
  JOBS=1762534,1849335 python scripts/upload_axis_eval_artifacts.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ENTITY = os.environ.get("WANDB_ENTITY", "aselghetany-nu")
PROJECT = os.environ.get("WANDB_PROJECT", "thesis_axes")
TRAIN_PROJECT = os.environ.get("WANDB_TRAIN_PROJECT", "block_qwen")
BASE = Path(
    os.environ.get(
        "AXIS_OUTPUT_ROOT",
        "/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen",
    )
)
DEFAULT_JOBS = [
    "1836796",
    "1762534",
    "1763301",
    "1773298",
    "1836800",
    "1849335",
    "1855537",
    "1849287",
    "1857323",
    "1836723",
    "1848844",
    "1836725",
    "1856101",
    "1856103",
    "1856105",
    "1857473",
    "1857278",
    "1857471",
    "1857475",
    "1857519",
    "1836811",
]


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


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _flatten(prefix: str, obj: Any, out: Dict[str, Any], max_depth: int = 4) -> None:
    if max_depth < 0 or obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, (int, float, bool, str)) or v is None:
                if isinstance(v, str) and len(v) > 200:
                    continue
                out[key] = v
            else:
                _flatten(key, v, out, max_depth - 1)
    elif isinstance(obj, list) and len(obj) <= 32 and all(
        isinstance(x, (int, float)) for x in obj
    ):
        out[prefix] = obj


def _run_dir(job: str) -> Optional[Path]:
    matches = sorted(BASE.glob(f"*_{job}"))
    return matches[0] if matches else None


def _train_run_id(run_dir: Path) -> Optional[str]:
    wandb_root = run_dir / "hydra" / "wandb"
    if not wandb_root.is_dir():
        return None
    offs = sorted(wandb_root.glob("offline-run-*"))
    if not offs:
        return None
    # offline-run-TIMESTAMP-RUNID
    name = offs[-1].name
    m = re.match(r"offline-run-\d+_\d+-(.+)$", name)
    return m.group(1) if m else None


def _collect_metric_files(run_dir: Path) -> List[Path]:
    candidates = [
        run_dir / "eval" / "gen_ppl_metrics.json",
        run_dir / "eval" / "gen_ppl_pair.json",
        run_dir / "eval" / "block_elbo_sweep.json",
        run_dir / "eval" / "eval_manifest.json",
        run_dir / "eval" / "samples.meta.json",
        run_dir / "eval" / "unifusion_hygiene" / "dual" / "gen_ppl_metrics_first_chunk.json",
        run_dir / "eval" / "unifusion_hygiene" / "dual" / "gen_ppl_metrics_full.json",
    ]
    # NFE sweep metrics if present
    nfe = run_dir / "eval" / "unifusion_hygiene" / "nfe_sweep"
    if nfe.is_dir():
        candidates.extend(sorted(nfe.glob("steps_*/gen_ppl_metrics.json")))
        candidates.extend(sorted(nfe.glob("steps_*/gen_ppl_pair.json")))
    # lm_eval SUMMARYs
    candidates.extend(sorted(run_dir.glob("lm_eval*/SUMMARY.json")))
    return [p for p in candidates if p.is_file()]


def _summary_scores(run_dir: Path) -> Dict[str, Any]:
    """Prefer hubmatch paper_acc SUMMARY for primary gsm/ife/mmlu."""
    out: Dict[str, Any] = {}
    preferred = [
        run_dir / "lm_eval_m2048_hubmatch_t1" / "SUMMARY.json",
        run_dir / "lm_eval_m2048_hubmatch_t1_mmlu_hubll" / "SUMMARY.json",
        run_dir / "lm_eval_m2048_baseline" / "SUMMARY.json",
    ]
    for path in preferred + sorted(run_dir.glob("lm_eval*/SUMMARY.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        suite = data.get("suite") or path.parent.name
        profile = data.get("decode_profile") or "unknown"
        tasks = data.get("tasks") or {}
        for task, info in tasks.items():
            if not isinstance(info, dict):
                continue
            score = info.get("score")
            if score is None:
                continue
            # percent for slide-comparable primary keys
            key = f"lm_eval/{suite}/{profile}/{task}"
            out[key] = float(score)
            if path.parent.name == "lm_eval_m2048_hubmatch_t1" or (
                suite in ("paper_acc", "paper-acc") and profile == "hubmatch"
            ):
                if task == "gsm8k":
                    out["gsm"] = float(score) * 100.0
                elif task == "ifeval":
                    out["ife"] = float(score) * 100.0
                elif task == "mmlu":
                    out["mmlu"] = float(score) * 100.0
    # hubll mmlu override
    hubll = run_dir / "lm_eval_m2048_hubmatch_t1_mmlu_hubll" / "SUMMARY.json"
    data = _read_json(hubll)
    if isinstance(data, dict):
        t = (data.get("tasks") or {}).get("mmlu") or {}
        if t.get("score") is not None:
            out["mmlu"] = float(t["score"]) * 100.0
            out["mmlu_source"] = "hubll"
    return out


def _gen_ppl_metrics(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    mapping = [
        ("gen_ppl", run_dir / "eval" / "gen_ppl_metrics.json"),
        ("gen_ppl_pair", run_dir / "eval" / "gen_ppl_pair.json"),
        (
            "gen_ppl_first_chunk",
            run_dir
            / "eval"
            / "unifusion_hygiene"
            / "dual"
            / "gen_ppl_metrics_first_chunk.json",
        ),
        (
            "gen_ppl_full",
            run_dir / "eval" / "unifusion_hygiene" / "dual" / "gen_ppl_metrics_full.json",
        ),
    ]
    for prefix, path in mapping:
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        for k in (
            "ppl",
            "avg_nll",
            "median_nll",
            "acc",
            "eos_rate",
            "full_length_rate",
            "unigram_entropy_mean",
            "unigram_entropy_median",
            "citeable",
            "first_chunk_only",
            "honesty_warning",
        ):
            if k in data and data[k] is not None:
                out[f"{prefix}/{k}"] = data[k]
        if prefix == "gen_ppl_first_chunk" and "ppl" in data:
            out["gen_ppl"] = data["ppl"]
            out["gen_ppl_H"] = data.get("unigram_entropy_mean")
        elif prefix == "gen_ppl" and "gen_ppl" not in out and "ppl" in data:
            out["gen_ppl"] = data["ppl"]
            out["gen_ppl_H"] = data.get("unigram_entropy_mean")
    elbo = _read_json(run_dir / "eval" / "block_elbo_sweep.json")
    if isinstance(elbo, dict):
        _flatten("elbo", elbo, out, max_depth=3)
    return out


def _sample_rows(samples_txt: Path, limit: int = 16) -> List[List[str]]:
    if not samples_txt.is_file():
        return []
    text = samples_txt.read_text(errors="replace")
    chunks = re.split(r"\n={5,}\n|\n-{5,}\n", text)
    rows = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        rows.append([str(i), chunk[:2000]])
        if len(rows) >= limit:
            break
    if not rows and text.strip():
        rows.append(["0", text[:2000]])
    return rows


def _upload_one(job: str, upload_ckpt: bool) -> str:
    import wandb

    run_dir = _run_dir(job)
    if run_dir is None:
        print(f"SKIP {job}: no run dir")
        return ""
    train_id = _train_run_id(run_dir)
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    metric_files = _collect_metric_files(run_dir)
    samples_txt = run_dir / "eval" / "samples.txt"
    samples_pt = run_dir / "eval" / "samples.pt"

    metrics: Dict[str, Any] = {
        "job_id": job,
        "run_dir": str(run_dir),
        "train_run_id": train_id,
        "checkpoint_path": str(ckpt) if ckpt.is_file() else None,
        "checkpoint_bytes": ckpt.stat().st_size if ckpt.is_file() else None,
    }
    metrics.update(_gen_ppl_metrics(run_dir))
    metrics.update(_summary_scores(run_dir))

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=f"eval_{job}",
        name=f"eval · {run_dir.name}",
        resume="allow",
        job_type="eval_artifacts",
        tags=["eval", f"job:{job}", run_dir.name.split("_")[0]],
        config={
            "job_id": job,
            "run_dir": str(run_dir),
            "train_project": TRAIN_PROJECT,
            "train_run_id": train_id,
            "train_url": (
                f"https://wandb.ai/{ENTITY}/{TRAIN_PROJECT}/runs/{train_id}"
                if train_id
                else None
            ),
        },
        settings=wandb.Settings(console="off", _disable_stats=True),
    )

    # Scalar log + summary
    scalar = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float, bool)) and not isinstance(v, bool) or isinstance(v, bool)
    }
    # keep bools; drop long strings from log step
    loggable = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 120)
    }
    run.log(loggable)
    for k, v in metrics.items():
        if isinstance(v, (int, float, bool, str)) and (
            not isinstance(v, str) or len(v) < 500
        ):
            run.summary[k] = v

    # Samples table
    rows = _sample_rows(samples_txt)
    if rows:
        table = wandb.Table(columns=["idx", "text"], data=rows)
        run.log({"samples/preview": table})

    # Eval bundle artifact (jsons + samples text/meta)
    art = wandb.Artifact(
        name=f"eval-bundle-{job}",
        type="eval",
        metadata={"job_id": job, "run_dir": str(run_dir)},
    )
    for path in metric_files:
        art.add_file(str(path), name=str(path.relative_to(run_dir)))
    if samples_txt.is_file():
        art.add_file(str(samples_txt), name="eval/samples.txt")
    if (run_dir / "eval" / "samples.meta.json").is_file():
        art.add_file(
            str(run_dir / "eval" / "samples.meta.json"),
            name="eval/samples.meta.json",
        )
    # samples.pt can be large; reference only
    if samples_pt.is_file():
        art.add_reference(f"file://{samples_pt}", name="eval/samples.pt")
    run.log_artifact(art)

    # Checkpoint artifact
    if ckpt.is_file():
        ckpt_art = wandb.Artifact(
            name=f"ckpt-{job}",
            type="model",
            metadata={
                "job_id": job,
                "path": str(ckpt),
                "bytes": ckpt.stat().st_size,
            },
        )
        if upload_ckpt:
            print(f"  uploading full ckpt ({ckpt.stat().st_size / 1e9:.1f}G) …")
            ckpt_art.add_file(str(ckpt), name="last.ckpt")
        else:
            ckpt_art.add_reference(f"file://{ckpt}", name="last.ckpt")
        run.log_artifact(ckpt_art)

    # Patch train run summary with eval pointers (loss already there from sync)
    if train_id:
        try:
            api = wandb.Api()
            train = api.run(f"{ENTITY}/{TRAIN_PROJECT}/{train_id}")
            train.summary["eval_run_id"] = f"eval_{job}"
            train.summary["eval_url"] = run.get_url()
            for k in ("gen_ppl", "gen_ppl_H", "gsm", "ife", "mmlu"):
                if k in metrics:
                    train.summary[k] = metrics[k]
            train.summary["checkpoint_path"] = str(ckpt) if ckpt.is_file() else None
            train.summary.update()
        except Exception as e:
            print(f"  WARN train summary patch failed: {e}")

    url = run.get_url()
    print(
        f"OK {job} metrics={len(loggable)} files={len(metric_files)} "
        f"samples={'yes' if rows else 'no'} ckpt={'ref' if ckpt.is_file() else 'no'} -> {url}"
    )
    run.finish()
    return url


def main() -> int:
    _load_key()
    jobs = [
        j.strip()
        for j in os.environ.get("JOBS", ",".join(DEFAULT_JOBS)).split(",")
        if j.strip()
    ]
    upload_ckpt = os.environ.get("UPLOAD_CKPT", "0") == "1"
    urls = []
    for job in jobs:
        urls.append(_upload_one(job, upload_ckpt=upload_ckpt))
    print(f"\nUploaded {sum(1 for u in urls if u)}/{len(jobs)} eval bundles")
    print(f"Project: https://wandb.ai/{ENTITY}/{PROJECT}")
    print(
        "Note: training loss curves live on synced runs under "
        f"https://wandb.ai/{ENTITY}/{TRAIN_PROJECT} "
        "(companion eval_* runs hold GenPPL/samples/lm_eval/ckpt refs)."
    )
    if not upload_ckpt:
        print("Checkpoints registered as file:// references (set UPLOAD_CKPT=1 to stream bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
