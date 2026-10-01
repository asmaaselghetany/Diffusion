#!/usr/bin/env python3
"""Attach eval tables/metrics/artifacts onto each training WandB run.

Organization matches ``src/discrete_diffusion/algorithms/base.py``:

  Charts (logged in-train on ``trainer/global_step``):
    ``trainer/loss``, ``train/{nll,bpd,ppl}``, ``val/{nll,bpd,ppl}``

  Tables (post-train attach on ``eval_attach_step``):
    ``val/samples``, ``samples@post_train``  (mirrors ``samples@global_stepN``)
    ``val/gen_ppl``, ``val/lm_eval``, ``val/elbo``, ``val/throughput``, ``val/kpis``

  Summary KPIs: ``val/gsm``, ``val/ife``, ``val/mmlu``, ``val/gen_ppl_score``, …

  Never log eval scalars as history (junk one-point charts).
  Cross-job compare stays on ``thesis_axes/scoreboard`` (``val/scoreboard``).

Usage:
  python scripts/upload_wandb_panels.py
  JOBS=1762534,1849335 python scripts/upload_wandb_panels.py
  UPLOAD_CKPT=1 python scripts/upload_wandb_panels.py
  SKIP_SCOREBOARD=1 JOBS=1762534 python scripts/upload_wandb_panels.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ENTITY = os.environ.get("WANDB_ENTITY", "aselghetany-nu")
SCOREBOARD_PROJECT = os.environ.get("WANDB_SCOREBOARD_PROJECT", "thesis_axes")
TRAIN_PROJECT = os.environ.get("WANDB_TRAIN_PROJECT", "block_qwen")
BASE = Path(
    os.environ.get(
        "AXIS_OUTPUT_ROOT",
        "/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen",
    )
)
SLURM_LOGS = Path(
    os.environ.get(
        "SLURM_LOG_ROOT",
        "/e/project1/scifi/elsayed3/Diffusion-new/slurm_logs",
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

# Human labels for the scoreboard / attach (job_id → display metadata).
# Display title format matches scripts/rename_wandb_runs.py:
#   {Paradigm} · {corruption} · {recipe} · {jobid}
META: Dict[str, Dict[str, str]] = {
    "1836796": dict(name="Native · masked · scratch · 1836796", paradigm="native", corruption="masked", components="none", recipe="scratch"),
    "1762534": dict(name="AR→block · masked · baseline · 1762534", paradigm="ar2block", corruption="masked", components="none", recipe="baseline"),
    "1763301": dict(name="AR→block · masked · math-mix (all5) · 1763301", paradigm="ar2block", corruption="masked", components="all5", recipe="math-mix (all5)"),
    "1773298": dict(name="AR→block · masked · chat-mix (all5) · 1773298", paradigm="ar2block", corruption="masked", components="all5", recipe="chat-mix (all5)"),
    "1836800": dict(name="AR→block · masked · mixed-SFT (all5) · 1836800", paradigm="ar2block", corruption="masked", components="all5", recipe="mixed-SFT (all5)"),
    "1849335": dict(name="AR→block · uniform · baseline · 1849335", paradigm="ar2block", corruption="uniform", components="none", recipe="baseline"),
    "1855537": dict(name="AR→block · masked · block-size mix · 1855537", paradigm="ar2block", corruption="masked", components="none", recipe="block-size mix"),
    "1849287": dict(name="AR→block · uniform · block-size mix · 1849287", paradigm="ar2block", corruption="uniform", components="none", recipe="block-size mix"),
    "1857323": dict(name="AR→full · n/a · matched AR SFT · 1857323", paradigm="ar_full", corruption="n/a", components="none", recipe="matched AR SFT"),
    "1836723": dict(name="Joint · masked · AR+causal · 1836723", paradigm="joint", corruption="masked", components="joint_ar+causal_clean", recipe="AR+causal"),
    "1848844": dict(name="Joint · uniform · AR+causal · 1848844", paradigm="joint", corruption="uniform", components="joint_ar+causal_clean", recipe="AR+causal"),
    "1836725": dict(name="Joint · masked · all5 + AR+causal · 1836725", paradigm="joint", corruption="masked", components="all5+joint", recipe="all5 + AR+causal"),
    "1856101": dict(name="AR→block · masked · +shift · 1856101", paradigm="ar2block", corruption="masked", components="shift", recipe="+shift"),
    "1856103": dict(name="AR→block · masked · +complementary · 1856103", paradigm="ar2block", corruption="masked", components="complementary", recipe="+complementary"),
    "1856105": dict(name="AR→block · masked · +shift+comp · 1856105", paradigm="ar2block", corruption="masked", components="shift+complementary", recipe="+shift+comp"),
    "1857473": dict(name="AR→block · masked · +intra-block anneal · 1857473", paradigm="ar2block", corruption="masked", components="intra_block_anneal", recipe="+intra-block anneal"),
    "1857278": dict(name="AR→block · uniform · +shift · 1857278", paradigm="ar2block", corruption="uniform", components="shift", recipe="+shift"),
    "1857471": dict(name="AR→block · uniform · +intra-block anneal · 1857471", paradigm="ar2block", corruption="uniform", components="intra_block_anneal", recipe="+intra-block anneal"),
    "1857475": dict(name="AR→block · hybrid · joint curriculum · 1857475", paradigm="ar2block", corruption="hybrid", components="anneals", recipe="joint curriculum"),
    "1857519": dict(name="AR→block · uniform · continue-FT from masked · 1857519", paradigm="ar2block", corruption="uniform", components="continue_ft", recipe="continue-FT from masked"),
    "1836811": dict(name="AR→block · hybrid · p(uniform)=0.1 · 1836811", paradigm="ar2block", corruption="hybrid", components="none", recipe="p(uniform)=0.1"),
}


def _load_key() -> None:
    if os.environ.get("WANDB_API_KEY"):
        pass
    else:
        candidates = [
            Path.home() / ".config/wandb/api_key",
        ]
        env_file = os.environ.get("WANDB_API_KEY_FILE", "").strip()
        if env_file:
            candidates.insert(0, Path(env_file).expanduser())
        for path in candidates:
            if path.is_file():
                os.environ["WANDB_API_KEY"] = path.read_text().strip()
                break
        else:
            raise SystemExit(
                "WANDB_API_KEY missing; export it or set WANDB_API_KEY_FILE / "
                "~/.config/wandb/api_key")
    # Keep WandB staging off $HOME (home quota is tight on this cluster).
    root = Path("/e/project1/scifi/elsayed3/Diffusion-new")
    cache = root / ".wandb_cache"
    data = root / ".wandb_data"
    cache.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache))
    os.environ.setdefault("WANDB_DATA_DIR", str(data))
    os.environ.setdefault("WANDB_DIR", str(root / "wandb"))


def _json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


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
    m = re.match(r"offline-run-\d+_\d+-(.+)$", offs[-1].name)
    return m.group(1) if m else None


def _pct(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(x) * 100.0


def _slurm_logs_for_job(job: str) -> List[Path]:
    out: List[Path] = []
    if not SLURM_LOGS.is_dir():
        return out
    for pat in (
        f"*_{job}.out",
        f"*_{job}.err",
    ):
        out.extend(sorted(SLURM_LOGS.glob(pat)))
    out.extend(sorted(SLURM_LOGS.glob(f"lever_*_{job}.out")))
    out.extend(sorted(SLURM_LOGS.glob(f"lever_*_{job}.err")))
    seen = set()
    uniq = []
    for p in out:
        if p not in seen and p.is_file() and p.stat().st_size < 80_000_000:
            seen.add(p)
            uniq.append(p)
    return uniq


def _collect_cell(job: str) -> Dict[str, Any]:
    run_dir = _run_dir(job)
    meta = META.get(job, dict(name=job, paradigm="?", corruption="?", components="?"))
    empty = {
        "job_id": job,
        "run_dir": None,
        "meta": meta,
        "kpis": {},
        "tables": {
            "gen_ppl": [],
            "elbo": [],
            "lm_eval": [],
            "throughput": [],
            "samples": [],
        },
        "files": [],
        "log_files": [],
        "ckpt": None,
        "train_run_id": None,
    }
    if run_dir is None:
        return empty

    train_id = _train_run_id(run_dir)
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    kpis: Dict[str, Any] = {
        "job_id": job,
        "display_name": meta["name"],
        "paradigm": meta["paradigm"],
        "corruption": meta["corruption"],
        "components": meta["components"],
        "run_dir": str(run_dir),
        "train_run_id": train_id,
        "train_url": (
            f"https://wandb.ai/{ENTITY}/{TRAIN_PROJECT}/runs/{train_id}"
            if train_id
            else None
        ),
        "checkpoint_path": str(ckpt) if ckpt.is_file() else None,
        "checkpoint_gb": round(ckpt.stat().st_size / 1e9, 2) if ckpt.is_file() else None,
    }

    gen_rows: List[List[Any]] = []
    sources = [
        ("eval_default", run_dir / "eval" / "gen_ppl_metrics.json"),
        ("pair", run_dir / "eval" / "gen_ppl_pair.json"),
        (
            "dual_first_chunk",
            run_dir
            / "eval"
            / "unifusion_hygiene"
            / "dual"
            / "gen_ppl_metrics_first_chunk.json",
        ),
        (
            "dual_full",
            run_dir / "eval" / "unifusion_hygiene" / "dual" / "gen_ppl_metrics_full.json",
        ),
    ]
    nfe = run_dir / "eval" / "unifusion_hygiene" / "nfe_sweep"
    if nfe.is_dir():
        for d in sorted(nfe.glob("steps_*")):
            sources.append((f"nfe_{d.name}", d / "gen_ppl_metrics.json"))

    for label, path in sources:
        data = _json(path)
        if not isinstance(data, dict) or "ppl" not in data:
            continue
        gen_rows.append(
            [
                label,
                data.get("ppl"),
                data.get("unigram_entropy_mean"),
                data.get("unigram_entropy_median"),
                data.get("eos_rate"),
                data.get("full_length_rate"),
                data.get("avg_nll"),
                data.get("median_nll"),
                data.get("citeable"),
                data.get("first_chunk_only"),
                data.get("honesty_warning"),
                str(path.relative_to(run_dir)),
            ]
        )
        if label == "dual_first_chunk":
            kpis["gen_ppl"] = data.get("ppl")
            kpis["gen_ppl_H"] = data.get("unigram_entropy_mean")
            kpis["gen_ppl_eos_rate"] = data.get("eos_rate")
            kpis["gen_ppl_flr"] = data.get("full_length_rate")
        elif label == "eval_default" and "gen_ppl" not in kpis:
            kpis["gen_ppl"] = data.get("ppl")
            kpis["gen_ppl_H"] = data.get("unigram_entropy_mean")

    elbo_rows: List[List[Any]] = []
    elbo = _json(run_dir / "eval" / "block_elbo_sweep.json")
    if isinstance(elbo, dict):
        for row in elbo.get("results") or []:
            if not isinstance(row, dict):
                continue
            elbo_rows.append(
                [
                    row.get("block_size"),
                    row.get("meter"),
                    row.get("mean_nll"),
                    row.get("bpd"),
                    row.get("ppl"),
                    row.get("tokens"),
                    row.get("batches"),
                ]
            )
            if row.get("block_size") == 32:
                kpis["elbo_ppl_bs32"] = row.get("ppl")
            if row.get("block_size") == 1:
                kpis["elbo_ppl_bs1"] = row.get("ppl")

    lm_rows: List[List[Any]] = []
    for summary_path in sorted(run_dir.glob("lm_eval*/SUMMARY.json")):
        data = _json(summary_path)
        if not isinstance(data, dict):
            continue
        suite = data.get("suite") or summary_path.parent.name
        profile = data.get("decode_profile")
        thr = data.get("unmask_threshold")
        for task, info in (data.get("tasks") or {}).items():
            if not isinstance(info, dict):
                continue
            score = info.get("score")
            if task.startswith("mmlu_") and task != "mmlu":
                continue
            lm_rows.append(
                [
                    summary_path.parent.name,
                    suite,
                    profile,
                    thr,
                    task,
                    score,
                    _pct(score) if isinstance(score, (int, float)) else None,
                    info.get("score_key"),
                ]
            )
        parent = summary_path.parent.name
        tasks = data.get("tasks") or {}
        if parent == "lm_eval_m2048_hubmatch_t1" or (
            suite in ("paper_acc", "paper-acc") and str(profile) == "hubmatch"
        ):
            if "gsm8k" in tasks and tasks["gsm8k"].get("score") is not None:
                kpis["gsm"] = _pct(tasks["gsm8k"]["score"])
            if "ifeval" in tasks and tasks["ifeval"].get("score") is not None:
                kpis["ife"] = _pct(tasks["ifeval"]["score"])
            if "mmlu" in tasks and tasks["mmlu"].get("score") is not None:
                kpis["mmlu"] = _pct(tasks["mmlu"]["score"])
        if parent == "lm_eval_m2048_hubmatch_t1_mmlu_hubll":
            t = tasks.get("mmlu") or {}
            if t.get("score") is not None:
                kpis["mmlu"] = _pct(t["score"])
                kpis["mmlu_meter"] = "hubll"
        if "code_chatfix" in parent:
            for code_task in ("humaneval", "humaneval_plus", "mbpp", "mbpp_plus"):
                if code_task in tasks and tasks[code_task].get("score") is not None:
                    kpis[code_task] = _pct(tasks[code_task]["score"])
        if parent == "lm_eval_m2048_baseline" and "gsm" not in kpis:
            if "gsm8k" in tasks and tasks["gsm8k"].get("score") is not None:
                kpis["gsm"] = _pct(tasks["gsm8k"]["score"])
            if "ifeval" in tasks and tasks["ifeval"].get("score") is not None:
                kpis["ife"] = _pct(tasks["ifeval"]["score"])

    tok_rows: List[List[Any]] = []
    for tok_path in sorted(run_dir.glob("lm_eval*/tok_s*.json")):
        data = _json(tok_path)
        if not isinstance(data, dict):
            continue
        tok_rows.append(
            [
                tok_path.parent.name,
                tok_path.name,
                data.get("decode_profile"),
                data.get("tok_s"),
                data.get("tokens_generated"),
                data.get("elapsed_s"),
                data.get("num_steps"),
                data.get("unmask_threshold"),
                data.get("hierarchical_kv"),
                data.get("use_block_cache"),
                data.get("source") or data.get("mode"),
            ]
        )
        if tok_path.name == "tok_s_lm_eval.json" and "hubmatch_t1" in tok_path.parent.name:
            if "code" not in tok_path.parent.name:
                kpis.setdefault("tok_s_lm_eval", data.get("tok_s"))

    sample_rows: List[List[Any]] = []
    samples_txt = run_dir / "eval" / "samples.txt"
    if samples_txt.is_file():
        text = samples_txt.read_text(errors="replace")
        chunks = re.split(r"\n={5,}\n|\n-{5,}\n", text)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            sample_rows.append([i, chunk[:2500]])
            if len(sample_rows) >= 24:
                break

    files: List[Path] = []
    for p in [
        run_dir / "eval" / "gen_ppl_metrics.json",
        run_dir / "eval" / "gen_ppl_pair.json",
        run_dir / "eval" / "block_elbo_sweep.json",
        run_dir / "eval" / "eval_manifest.json",
        run_dir / "eval" / "samples.meta.json",
        run_dir / "eval" / "samples.txt",
        run_dir
        / "eval"
        / "unifusion_hygiene"
        / "dual"
        / "gen_ppl_metrics_first_chunk.json",
        run_dir / "eval" / "unifusion_hygiene" / "dual" / "gen_ppl_metrics_full.json",
    ]:
        if p.is_file():
            files.append(p)
    if nfe.is_dir():
        files.extend(sorted(nfe.glob("steps_*/gen_ppl_metrics.json")))
    files.extend(sorted(run_dir.glob("lm_eval*/SUMMARY.json")))
    files.extend(sorted(run_dir.glob("lm_eval*/SUMMARY.md")))
    files.extend(sorted(run_dir.glob("lm_eval*/tok_s*.json")))
    for p in [
        run_dir / "hydra" / ".hydra" / "config.yaml",
        run_dir / "hydra" / ".hydra" / "overrides.yaml",
    ]:
        if p.is_file():
            files.append(p)

    log_files = _slurm_logs_for_job(job)
    log_files.extend(sorted(run_dir.glob("lm_eval*/lm_eval.log")))

    return {
        "job_id": job,
        "run_dir": run_dir,
        "meta": meta,
        "kpis": kpis,
        "tables": {
            "gen_ppl": gen_rows,
            "elbo": elbo_rows,
            "lm_eval": lm_rows,
            "throughput": tok_rows,
            "samples": sample_rows,
        },
        "files": files,
        "log_files": log_files,
        "ckpt": ckpt if ckpt.is_file() else None,
        "train_run_id": train_id,
    }


def _head_tail_text(path: Path, head: int = 2500, tail: int = 2500) -> str:
    """Read a large log without loading everything if possible."""
    try:
        raw = path.read_text(errors="replace")
    except Exception as e:
        return f"[failed to read {path}: {e}]\n"
    lines = raw.splitlines()
    if len(lines) <= head + tail:
        return raw
    omitted = len(lines) - head - tail
    parts = (
        lines[:head]
        + [f"\n… [{omitted} lines omitted] …\n"]
        + lines[-tail:]
    )
    return "\n".join(parts) + "\n"


def _build_output_log(job: str, log_files: List[Path]) -> str:
    """Console-style log for WandB Logs tab (output.log)."""
    chunks: List[str] = [
        f"# Console log reconstructed for job {job}\n",
        "# Training used WANDB_CONSOLE=off (avoid tqdm 429s); "
        "slurm stdout/stderr + lm_eval logs are mirrored here.\n\n",
    ]
    # Prefer train .out first, then .err, then lm_eval logs.
    ordered = sorted(
        log_files,
        key=lambda p: (
            0 if p.suffix == ".out" and "lever" in p.name else
            1 if p.suffix == ".err" else
            2 if p.name == "lm_eval.log" else 3,
            str(p),
        ),
    )
    for path in ordered:
        chunks.append("=" * 72 + "\n")
        chunks.append(f"# FILE: {path}\n")
        chunks.append("=" * 72 + "\n")
        chunks.append(_head_tail_text(path))
        chunks.append("\n")
    return "".join(chunks) if ordered else "# No local log files found.\n"


def _publish_run_logs(run, job: str, log_files: List[Path]) -> int:
    """Put logs where the UI expects them: Files + Logs (output.log)."""
    import tempfile

    if not log_files:
        return 0
    run_dir = Path(run.dir)
    files_dir = run_dir  # wandb syncs files written under run.dir
    n = 0
    # Individual copies under logs/
    log_subdir = files_dir / "logs"
    log_subdir.mkdir(parents=True, exist_ok=True)
    for path in log_files:
        dest = log_subdir / path.name
        try:
            # Cap per-file copy at 32MB for the Files browser
            data = path.read_bytes()
            if len(data) > 32_000_000:
                text = _head_tail_text(path, head=4000, tail=4000)
                dest.write_text(text, errors="replace")
            else:
                dest.write_bytes(data)
            n += 1
        except Exception as e:
            print(f"  WARN copy log {path.name}: {e}")

    # Combined output.log → WandB Logs tab
    output_log = files_dir / "output.log"
    output_log.write_text(_build_output_log(job, log_files), errors="replace")
    # Force upload of these files into the run
    try:
        import wandb as _wandb

        _wandb.save(str(output_log), base_path=str(files_dir), policy="now")
        for p in log_subdir.glob("*"):
            _wandb.save(str(p), base_path=str(files_dir), policy="now")
    except Exception as e:
        print(f"  WARN wandb.save logs: {e}")
    run.summary["logs_published"] = True
    run.summary["logs_n_files"] = n
    run.summary["logs_output"] = "output.log"
    return n


def _build_media(wandb_mod, tables: Dict[str, Any], kpis: Dict[str, Any]) -> Dict[str, Any]:
    """Media keys match ``algorithms/base.py`` WandB template.

    Charts (already logged in-train): ``trainer/loss``, ``train/*``, ``val/nll|bpd|ppl``.
    Tables (post-train attach, same UI family):
      - ``val/samples`` + ``samples@post_train``  (mirrors ``val/samples`` / ``samples@global_stepN``)
      - ``val/gen_ppl``, ``val/lm_eval``, ``val/elbo``, ``val/throughput``, ``val/kpis``
    """
    media: Dict[str, Any] = {}
    if tables["gen_ppl"]:
        media["val/gen_ppl"] = wandb_mod.Table(
            columns=[
                "source",
                "ppl",
                "H_mean",
                "H_median",
                "eos_rate",
                "full_length_rate",
                "avg_nll",
                "median_nll",
                "citeable",
                "first_chunk_only",
                "honesty_warning",
                "path",
            ],
            data=tables["gen_ppl"],
        )
    if tables["elbo"]:
        media["val/elbo"] = wandb_mod.Table(
            columns=["block_size", "meter", "mean_nll", "bpd", "ppl", "tokens", "batches"],
            data=tables["elbo"],
        )
    if tables["lm_eval"]:
        media["val/lm_eval"] = wandb_mod.Table(
            columns=[
                "suite_dir",
                "suite",
                "decode_profile",
                "threshold",
                "task",
                "score",
                "score_pct",
                "score_key",
            ],
            data=tables["lm_eval"],
        )
    if tables["throughput"]:
        media["val/throughput"] = wandb_mod.Table(
            columns=[
                "suite_dir",
                "file",
                "decode_profile",
                "tok_s",
                "tokens_generated",
                "elapsed_s",
                "num_steps",
                "threshold",
                "hierarchical_kv",
                "block_cache",
                "source",
            ],
            data=tables["throughput"],
        )
    if tables["samples"]:
        # Match base.py: column name + dual keys (stable alias + step-tagged).
        rows = [[row[1]] for row in tables["samples"]]
        media["val/samples"] = wandb_mod.Table(columns=["Generated Samples"], data=rows)
        media["samples@post_train"] = wandb_mod.Table(
            columns=["Generated Samples"], data=rows
        )
    media["val/kpis"] = wandb_mod.Table(
        columns=list(kpis.keys()),
        data=[[kpis.get(c) for c in kpis.keys()]],
    )
    return media


def _attach_eval_to_train_run(cell: Dict[str, Any], upload_ckpt: bool) -> str:
    import wandb

    job = cell["job_id"]
    meta = cell["meta"]
    kpis = cell["kpis"]
    tables = cell["tables"]
    run_dir: Optional[Path] = cell["run_dir"]
    train_id = cell["train_run_id"]

    if not train_id:
        print(f"SKIP {job}: no train wandb id (offline-run missing)")
        return ""

    run = wandb.init(
        entity=ENTITY,
        project=TRAIN_PROJECT,
        id=train_id,
        name=meta.get("name"),  # clear display title; id stays stable for resume
        resume="allow",
        # wrap so reconstructed logs land in the Logs tab (not only Artifacts).
        settings=wandb.Settings(console="wrap", _disable_stats=True),
    )
    if meta.get("name"):
        # Force display name even when resume restores an old lever_* title.
        run.name = meta["name"]

    # Separate step axis so post-train tables do not distort train loss charts.
    # In-train charts stay on trainer/global_step (trainer/loss, train/*, val/*).
    wandb.define_metric("trainer/global_step")
    wandb.define_metric("train/*", step_metric="trainer/global_step")
    wandb.define_metric("val/nll", step_metric="trainer/global_step")
    wandb.define_metric("val/bpd", step_metric="trainer/global_step")
    wandb.define_metric("val/ppl", step_metric="trainer/global_step")
    wandb.define_metric("eval_attach_step")
    wandb.define_metric("val/gen_ppl", step_metric="eval_attach_step")
    wandb.define_metric("val/lm_eval", step_metric="eval_attach_step")
    wandb.define_metric("val/elbo", step_metric="eval_attach_step")
    wandb.define_metric("val/throughput", step_metric="eval_attach_step")
    wandb.define_metric("val/samples", step_metric="eval_attach_step")
    wandb.define_metric("val/kpis", step_metric="eval_attach_step")
    wandb.define_metric("samples@post_train", step_metric="eval_attach_step")

    # Headline KPIs in summary under val/ (same family as in-train val/*).
    # Avoid colliding with table keys: val/gen_ppl, val/lm_eval, val/elbo are tables.
    summary_map = {
        "gsm": "val/gsm",
        "ife": "val/ife",
        "mmlu": "val/mmlu",
        "mmlu_meter": "val/mmlu_meter",
        "gen_ppl": "val/gen_ppl_score",
        "gen_ppl_H": "val/gen_ppl_H",
        "gen_ppl_eos_rate": "val/gen_ppl_eos_rate",
        "gen_ppl_flr": "val/gen_ppl_flr",
        "elbo_ppl_bs1": "val/elbo_ppl_bs1",
        "elbo_ppl_bs32": "val/elbo_ppl_bs32",
        "humaneval": "val/humaneval",
        "humaneval_plus": "val/humaneval_plus",
        "mbpp": "val/mbpp",
        "mbpp_plus": "val/mbpp_plus",
        "tok_s_lm_eval": "val/tok_s",
        "checkpoint_path": "checkpoint_path",
        "checkpoint_gb": "checkpoint_gb",
        "display_name": "display_name",
        "paradigm": "paradigm",
        "corruption": "corruption",
        "components": "components",
        "job_id": "job_id",
        "run_dir": "run_dir",
    }
    for src_key, dst_key in summary_map.items():
        if src_key in kpis and kpis[src_key] is not None:
            run.summary[dst_key] = kpis[src_key]
            # Keep short aliases too for filters / scoreboard joins.
            if src_key in (
                "gsm",
                "ife",
                "mmlu",
                "gen_ppl",
                "gen_ppl_H",
                "humaneval",
                "mbpp_plus",
                "tok_s_lm_eval",
            ):
                run.summary[src_key] = kpis[src_key]
    run.summary["eval_attached"] = True
    run.summary["eval_layout"] = "src_template_val_tables"

    run.config.update(
        {
            "thesis/job_id": job,
            "thesis/paradigm": meta["paradigm"],
            "thesis/corruption": meta["corruption"],
            "thesis/components": meta["components"],
            "thesis/display_name": meta["name"],
        },
        allow_val_change=True,
    )

    # WandB auto-names table artifacts run-{id}-…; '+' in lever tags breaks that.
    id_safe = bool(re.fullmatch(r"[A-Za-z0-9._-]+", train_id))
    media = _build_media(wandb, tables, kpis)
    if media and id_safe:
        run.log({"eval_attach_step": 1, **media})
    elif media and not id_safe:
        print(
            f"  NOTE {job}: train id has unsafe chars; "
            "tables go into eval-bundle JSON (not Media panels)"
        )

    # Bundle: tables JSON always; raw eval files optional (home/project quota).
    include_raw = os.environ.get("INCLUDE_RAW_FILES", "0") == "1"
    max_mb = float(os.environ.get("MAX_ARTIFACT_MB", "8"))
    if run_dir is not None and (cell["files"] or any(tables.values())):
        try:
            import tempfile

            art = wandb.Artifact(
                name=f"eval-bundle-{job}",
                type="eval",
                metadata={"job_id": job, "attached_to": train_id},
            )
            if include_raw:
                for path in cell["files"]:
                    try:
                        if path.stat().st_size > max_mb * 1e6:
                            continue
                        art.add_file(str(path), name=str(path.relative_to(run_dir)))
                    except Exception as e:
                        print(f"  WARN skip raw {path.name}: {e}")
            samples_pt = run_dir / "eval" / "samples.pt"
            if samples_pt.is_file():
                try:
                    art.add_reference(f"file://{samples_pt}", name="eval/samples.pt")
                except Exception:
                    pass
            with tempfile.TemporaryDirectory(prefix=f"wandb_tables_{job}_") as tmp:
                tdir = Path(tmp)
                specs = {
                    "gen_ppl": (
                        [
                            "source",
                            "ppl",
                            "H_mean",
                            "H_median",
                            "eos_rate",
                            "full_length_rate",
                            "avg_nll",
                            "median_nll",
                            "citeable",
                            "first_chunk_only",
                            "honesty_warning",
                            "path",
                        ],
                        tables["gen_ppl"],
                    ),
                    "elbo_by_block_size": (
                        [
                            "block_size",
                            "meter",
                            "mean_nll",
                            "bpd",
                            "ppl",
                            "tokens",
                            "batches",
                        ],
                        tables["elbo"],
                    ),
                    "lm_eval": (
                        [
                            "suite_dir",
                            "suite",
                            "decode_profile",
                            "threshold",
                            "task",
                            "score",
                            "score_pct",
                            "score_key",
                        ],
                        tables["lm_eval"],
                    ),
                    "throughput": (
                        [
                            "suite_dir",
                            "file",
                            "decode_profile",
                            "tok_s",
                            "tokens_generated",
                            "elapsed_s",
                            "num_steps",
                            "threshold",
                            "hierarchical_kv",
                            "block_cache",
                            "source",
                        ],
                        tables["throughput"],
                    ),
                    "samples": (["idx", "text"], tables["samples"]),
                    "kpis": (
                        list(kpis.keys()),
                        [[kpis.get(c) for c in kpis.keys()]],
                    ),
                }
                for name, (cols, rows) in specs.items():
                    if not rows:
                        continue
                    out = tdir / f"{name}.json"
                    out.write_text(
                        json.dumps(
                            {"columns": cols, "data": rows}, indent=2, default=str
                        )
                    )
                    art.add_file(str(out), name=f"tables/{name}.json")
            run.log_artifact(art, aliases=["latest", f"job-{job}"])
            run.summary["eval_bundle"] = f"eval-bundle-{job}:latest"
        except OSError as e:
            print(f"  WARN eval-bundle skipped ({e})")
        except Exception as e:
            print(f"  WARN eval-bundle skipped: {e}")

    if cell["log_files"]:
        n_logs = _publish_run_logs(run, job, cell["log_files"])
        # Also stream a truncated console mirror so the Logs tab is not empty.
        console_text = _build_output_log(job, cell["log_files"])
        if len(console_text) > 400_000:
            console_text = (
                console_text[:200_000]
                + "\n\n… [truncated for Logs tab; full log in Files/output.log "
                "and Artifacts/logs-*] …\n\n"
                + console_text[-200_000:]
            )
        print(console_text, flush=True)
        try:
            logs = wandb.Artifact(
                name=f"logs-{job}",
                type="logs",
                metadata={"job_id": job},
            )
            for path in cell["log_files"]:
                try:
                    if path.stat().st_size > max_mb * 1e6:
                        continue
                    logs.add_file(str(path), name=path.name)
                except Exception:
                    pass
            run.log_artifact(logs, aliases=["latest"])
        except Exception as e:
            print(f"  WARN logs artifact skipped: {e}")
        print(f"  published {n_logs} log files + output.log")

    if cell["ckpt"] is not None:
        try:
            ckpt_art = wandb.Artifact(
                name=f"ckpt-{job}",
                type="model",
                metadata={"job_id": job, "path": str(cell["ckpt"])},
            )
            if upload_ckpt:
                ckpt_art.add_file(str(cell["ckpt"]), name="last.ckpt")
            else:
                ckpt_art.add_reference(f"file://{cell['ckpt']}", name="last.ckpt")
            run.log_artifact(ckpt_art, aliases=["latest", "last"])
        except Exception as e:
            print(f"  WARN ckpt artifact skipped: {e}")

    url = run.get_url()
    print(
        f"OK attach {train_id} job={job} "
        f"gen={len(tables['gen_ppl'])} lm={len(tables['lm_eval'])} "
        f"elbo={len(tables['elbo'])} tok={len(tables['throughput'])} "
        f"samples={len(tables['samples'])} logs={len(cell['log_files'])} -> {url}"
    )
    run.finish()
    return url


def _upload_scoreboard(cells: List[Dict[str, Any]]) -> str:
    import wandb

    run = wandb.init(
        entity=ENTITY,
        project=SCOREBOARD_PROJECT,
        id="scoreboard",
        name="Thesis scoreboard (tables)",
        resume="allow",
        job_type="scoreboard",
        tags=["scoreboard", "table_first"],
        config={
            "layout": "src_template",
            "note": (
                "Charts: trainer/loss + train/* + val/{nll,bpd,ppl} on each train run. "
                "Tables: val/samples, val/gen_ppl, val/lm_eval, val/elbo, val/throughput "
                "(same family as algorithms/base.py). This board compares cells."
            ),
            "n_cells": len(cells),
        },
        settings=wandb.Settings(console="off", _disable_stats=True),
    )

    cols = [
        "job_id",
        "name",
        "paradigm",
        "corruption",
        "components",
        "gsm",
        "ife",
        "mmlu",
        "gen_ppl",
        "gen_ppl_H",
        "elbo_ppl_bs1",
        "elbo_ppl_bs32",
        "humaneval",
        "mbpp_plus",
        "tok_s_lm_eval",
        "train_url",
        "checkpoint_path",
    ]
    rows = []
    for cell in cells:
        k = cell["kpis"]
        m = cell["meta"]
        rows.append(
            [
                cell["job_id"],
                m["name"],
                m["paradigm"],
                m["corruption"],
                m["components"],
                k.get("gsm"),
                k.get("ife"),
                k.get("mmlu"),
                k.get("gen_ppl"),
                k.get("gen_ppl_H"),
                k.get("elbo_ppl_bs1"),
                k.get("elbo_ppl_bs32"),
                k.get("humaneval"),
                k.get("mbpp_plus"),
                k.get("tok_s_lm_eval"),
                k.get("train_url"),
                k.get("checkpoint_path"),
            ]
        )

    board = wandb.Table(columns=cols, data=rows)
    # Scoreboard tables under val/ for the same panel organization.
    payload: Dict[str, Any] = {
        "val/scoreboard": board,
        "tables/scoreboard": board,  # keep alias for existing dashboards
    }

    def _bar(metric: str, title: str):
        sub_data = [
            [r[cols.index("name")], r[cols.index(metric)]]
            for r in rows
            if r[cols.index(metric)] is not None
        ]
        if not sub_data:
            return None
        t = wandb.Table(columns=["name", metric], data=sub_data)
        return wandb.plot.bar(t, "name", metric, title=title)

    for metric, title in (
        ("gsm", "GSM8K (%)"),
        ("ife", "IFEval (%)"),
        ("mmlu", "MMLU (%)"),
        ("gen_ppl", "GenPPL (first_chunk)"),
        ("gen_ppl_H", "Unigram H̄"),
    ):
        plot = _bar(metric, title)
        if plot is not None:
            payload[f"plots/{metric}"] = plot

    run.log(payload)
    run.summary["n_cells"] = len(cells)
    run.summary["n_with_gsm"] = sum(1 for r in rows if r[cols.index("gsm")] is not None)
    run.summary["n_with_gen_ppl"] = sum(
        1 for r in rows if r[cols.index("gen_ppl")] is not None
    )

    url = run.get_url()
    print(f"OK scoreboard ({len(rows)} rows) -> {url}")
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
    skip_board = os.environ.get("SKIP_SCOREBOARD", "0") == "1"
    logs_only = os.environ.get("LOGS_ONLY", "0") == "1"

    cells = [_collect_cell(j) for j in jobs]
    if logs_only:
        import wandb

        print(f"LOGS_ONLY: publishing console logs for {len(cells)} train runs")
        for cell in cells:
            train_id = cell["train_run_id"]
            job = cell["job_id"]
            if not train_id:
                print(f"SKIP {job}: no train id")
                continue
            run = wandb.init(
                entity=ENTITY,
                project=TRAIN_PROJECT,
                id=train_id,
                resume="allow",
                settings=wandb.Settings(console="wrap", _disable_stats=True),
            )
            n = _publish_run_logs(run, job, cell["log_files"])
            console_text = _build_output_log(job, cell["log_files"])
            if len(console_text) > 400_000:
                console_text = (
                    console_text[:200_000]
                    + "\n\n… [truncated for Logs tab; full in Files/output.log] …\n\n"
                    + console_text[-200_000:]
                )
            print(console_text, flush=True)
            if cell["log_files"]:
                logs = wandb.Artifact(
                    name=f"logs-{job}",
                    type="logs",
                    metadata={"job_id": job},
                )
                for path in cell["log_files"]:
                    try:
                        logs.add_file(str(path), name=path.name)
                    except Exception:
                        pass
                run.log_artifact(logs, aliases=["latest"])
            print(f"OK logs {train_id} job={job} n_files={n} -> {run.get_url()}")
            run.finish()
        return 0

    print(f"Collected {len(cells)} cells; attaching eval onto {TRAIN_PROJECT} train runs")

    urls = []
    for cell in cells:
        urls.append(_attach_eval_to_train_run(cell, upload_ckpt=upload_ckpt))

    if not skip_board:
        _upload_scoreboard(cells)

    n_ok = sum(1 for u in urls if u)
    print(
        f"\nAttached eval to {n_ok}/{len(jobs)} train runs on "
        f"https://wandb.ai/{ENTITY}/{TRAIN_PROJECT}\n"
        "Organization (src template):\n"
        "  Charts  → trainer/loss, train/*, val/nll|bpd|ppl\n"
        "  Tables  → val/samples, samples@post_train, val/gen_ppl, "
        "val/lm_eval, val/elbo, val/throughput, val/kpis\n"
        "  Summary → val/gsm, val/ife, val/mmlu, val/gen_ppl, …\n"
        f"Cross-job board: https://wandb.ai/{ENTITY}/{SCOREBOARD_PROJECT}/runs/scoreboard"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
