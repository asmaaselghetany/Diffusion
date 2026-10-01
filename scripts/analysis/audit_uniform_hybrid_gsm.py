#!/usr/bin/env python3
"""Deep audit: all bad uniform/hybrid Instruct cells vs C0."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path("/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen")
DOC = Path("/e/project1/scifi/elsayed3/Diffusion-new/docs/research")

CELLS = {
    "1762534": ("C0 masked baseline", "masked", "control"),
    "1849335": ("U0 uniform baseline", "uniform", "bad"),
    "1857278": ("U0+shift", "uniform", "bad"),
    "1857471": ("U0+anneal", "uniform", "bad"),
    "1857519": ("U0 from C0", "uniform", "bad"),
    "1849287": ("xfer block-mix U", "uniform", "bad"),
    "1848844": ("U2 joint AR", "uniform", "bad"),
    "1855541": ("xfer bg-mix U", "uniform", "bad"),
    "1857475": ("hybrid B4 anneals", "hybrid", "bad"),
    "1836811": ("hybrid p=0.1", "hybrid", "missing"),
}


def run_dir(job: str) -> Path | None:
    ms = sorted(ROOT.glob(f"*_{job}"))
    return ms[0] if ms else None


def load_cfg(rd: Path):
    p = rd / "hydra/.hydra/config.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else None


def wandb_val(rd: Path) -> dict:
    summary = rd / "hydra/wandb/latest-run/files/wandb-summary.json"
    if not summary.exists():
        wand = rd / "hydra/wandb"
        cands = list(wand.glob("run-*/files/wandb-summary.json")) if wand.exists() else []
        summary = cands[-1] if cands else None
    if not summary or not Path(summary).exists():
        return {}
    d = json.loads(Path(summary).read_text())
    return {k: d[k] for k in ("val/ppl", "val/nll", "val/bpd", "train/ppl", "trainer/loss") if k in d}


def best_scores(rd: Path) -> dict:
    out: dict = {}
    for s in rd.rglob("SUMMARY.json"):
        if "hf_" in str(s):
            continue
        try:
            d = json.loads(s.read_text())
        except Exception:
            continue
        tasks = d.get("tasks") or {}
        meta = {
            "dir": s.parent.name,
            "profile": d.get("decode_profile"),
            "thr": d.get("unmask_threshold"),
            "greedy": d.get("force_greedy"),
            "suite": d.get("suite"),
            "mtime": s.stat().st_mtime,
        }
        for task, key in [
            ("gsm", "gsm8k"),
            ("ife", "ifeval"),
            ("mmlu", "mmlu"),
            ("mmlu_gen", "mmlu_generative"),
        ]:
            if key not in tasks or tasks[key].get("score") is None:
                continue
            v = float(tasks[key]["score"])
            if v <= 1.5:
                v *= 100
            rank = 0
            p = str(s)
            if "paper_gen" in p:
                rank += 3
            if "gsm8k" in p:
                rank += 2
            if "m2048" in p:
                rank += 1
            cur = out.get(task)
            if cur is None or rank > cur["rank"] or (rank == cur["rank"] and meta["mtime"] > cur["mtime"]):
                out[task] = {"score": v, "rank": rank, **meta}
    return out


def gen_ppl(rd: Path):
    best = None
    for p in rd.rglob("gen_ppl_metrics*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        score = (
            (2 if "unifusion_hygiene/dual" in str(p) else 0)
            + (1 if "first_chunk" in p.name or d.get("first_chunk_only") else 0)
            + (1 if d.get("citeable", True) else 0)
        )
        rec = {
            "ppl": d.get("ppl"),
            "H": d.get("unigram_entropy_mean"),
            "citeable": d.get("citeable"),
            "eos": d.get("eos_rate"),
            "warn": d.get("honesty_warning"),
            "path": str(p),
            "score": score,
        }
        if best is None or rec["score"] > best["score"]:
            best = rec
    return best


def analyze_log_answers(log_path):
    if not log_path or not Path(log_path).exists():
        return None
    text = Path(log_path).read_text(errors="replace")
    blocks = text.split("====================")
    answers = []
    for b in blocks:
        m = re.search(r"(?ms)^answer: ([\s\S]+)", b)
        if not m:
            continue
        answers.append(m.group(1).strip())
    if not answers:
        return {"n_answers_logged": 0}
    broken = sum(
        1
        for a in answers
        if re.search(r"let'?s let|KCalculator|alphabet|\\\\frac\s*\(|Max's's", a)
    )
    has_num = sum(1 for a in answers if re.search(r"\d", a))
    greedy_warn = text.count("greedy=true is invalid for uniform") + text.count(
        "invalid for uniform reverse"
    )
    lens = [len(a) for a in answers]
    return {
        "n_answers_logged": len(answers),
        "mean_len": sum(lens) / len(lens),
        "broken_markers": broken,
        "has_digit": has_num,
        "im_start_count": text.count("<|im_start|>"),
        "greedy_uniform_warns": greedy_warn,
        "hash_answers": text.count("####"),
        "sample0": answers[0][:220].replace("\n", " | "),
        "sample1": answers[1][:220].replace("\n", " | ") if len(answers) > 1 else None,
    }


def fmt(x, nd=1):
    return f"{x:.{nd}f}" if isinstance(x, float) else "—"


rows = []
for job, (label, corr, tag) in CELLS.items():
    rd = run_dir(job)
    rec = {
        "job": job,
        "label": label,
        "corruption": corr,
        "tag": tag,
        "run": str(rd) if rd else None,
    }
    if not rd:
        rec["error"] = "missing run dir"
        rows.append(rec)
        continue
    cfg = load_cfg(rd)
    if cfg:
        algo = cfg.get("algo") or {}
        samp = cfg.get("sampling") or algo.get("sampling") or {}
        rec.update(
            {
                "forward": algo.get("forward_process_name"),
                "loss_type": algo.get("loss_type"),
                "loss_weighting": algo.get("loss_weighting"),
                "ignore_bos": algo.get("ignore_bos"),
                "shift": algo.get("shift_loss_targets"),
                "comp": algo.get("complementary_masks"),
                "joint_ar": algo.get("joint_ar_alpha"),
                "anneal": algo.get("intra_block_attn_anneal_steps"),
                "kernel_anneal": algo.get("kernel_anneal_steps"),
                "hybrid_p": algo.get("hybrid_p_uniform"),
                "block_mix": algo.get("block_size_mixture") or algo.get("block_weights"),
                "max_steps": (cfg.get("trainer") or {}).get("max_steps"),
                "gbs": (cfg.get("loader") or {}).get("global_batch_size"),
                "block_size": cfg.get("block_size"),
                "load_pretrained": (cfg.get("model") or {}).get("load_pretrained"),
                "samp_steps": samp.get("steps"),
                "samp_greedy": samp.get("greedy"),
                "samp_thr": samp.get("unmask_threshold"),
                "samp_dual": samp.get("use_block_cache"),
            }
        )
    rec["val"] = wandb_val(rd)
    rec["scores"] = best_scores(rd)
    rec["gen_ppl"] = gen_ppl(rd)
    logs = sorted(rd.rglob("lm_eval.log"), key=lambda p: p.stat().st_size, reverse=True)
    gsm_logs = [p for p in logs if any(x in str(p) for x in ("gsm", "paper_gen", "m2048"))]
    log = gsm_logs[0] if gsm_logs else (logs[0] if logs else None)
    rec["log_audit"] = analyze_log_answers(log)
    rec["log_path"] = str(log) if log else None
    rec["has_last"] = (rd / "checkpoints/last.ckpt").is_file()
    rows.append(rec)

lines = [
    "# Audit: collapsed Instruct GSM on uniform/hybrid (2026-09-20)",
    "",
    "Question: is ~1–7% GSM normal for our uniform/hybrid conversion cells, or a systematic bug?",
    "",
    f"Generated: {datetime.now().isoformat(timespec='seconds')}",
    "",
    "## Verdict (short)",
    "",
    "Training looks **healthy** across these cells (val/ppl ~4, GenPPL often citeable with high H̄).",
    "Instruct **generative GSM is systematically broken** on every uniform conversion we measured —",
    "same failure mode as U0 (fluent garbage CoT, not empty EOS collapse).",
    "This is **not** explained by a single missing SUMMARY or one bad job.",
    "Masked C0 control remains ~62–64% GSM under DualCache/hubmatch.",
    "",
    "## Scoreboard",
    "",
    "| Cell | Job | Corr | val/ppl | GenPPL (H̄) | GSM | IFE | Eval profile |",
    "|------|-----|------|--------:|------------:|----:|----:|--------------|",
]
for r in rows:
    valp = (r.get("val") or {}).get("val/ppl")
    gp = r.get("gen_ppl") or {}
    sc = r.get("scores") or {}
    gsm = (sc.get("gsm") or {}).get("score")
    ife = (sc.get("ife") or {}).get("score")
    gmeta = sc.get("gsm") or sc.get("ife") or {}
    if gp.get("ppl") is not None and gp.get("H") is not None:
        gph = f"{gp['ppl']:.0f} ({gp['H']:.2f})"
    elif gp.get("ppl") is not None:
        gph = f"{gp['ppl']:.0f}"
    else:
        gph = "—"
    lines.append(
        f"| {r['label']} | `{r['job']}` | {r['corruption']} | "
        f"{fmt(valp, 2)} | {gph} | {fmt(gsm)} | {fmt(ife)} | "
        f"{gmeta.get('profile') or '—'} thr={gmeta.get('thr')} g={gmeta.get('greedy')} |"
    )

lines += [
    "",
    "## Train knobs",
    "",
    "| Cell | forward | steps | gbs | bs | shift | joint | anneal | loss_w | ignore_bos | pretrained |",
    "|------|---------|------:|----:|---:|------:|------:|-------:|--------|------------|------------|",
]
for r in rows:
    lines.append(
        f"| {r['label']} | {r.get('forward')} | {r.get('max_steps')} | {r.get('gbs')} | "
        f"{r.get('block_size')} | {r.get('shift')} | {r.get('joint_ar')} | {r.get('anneal')} | "
        f"{r.get('loss_weighting')} | {r.get('ignore_bos')} | {r.get('load_pretrained')} |"
    )

lines += [
    "",
    "## Generation forensics (lm_eval logs)",
    "",
    "| Cell | n answers | greedy→ancestral warns | broken markers | sample |",
    "|------|----------:|-----------------------:|---------------:|--------|",
]
for r in rows:
    la = r.get("log_audit") or {}
    if not la or la.get("n_answers_logged", 0) == 0:
        lines.append(f"| {r['label']} | — | — | — | (no log) |")
        continue
    samp = (la.get("sample0") or "").replace("|", "/")
    lines.append(
        f"| {r['label']} | {la.get('n_answers_logged')} | {la.get('greedy_uniform_warns')} | "
        f"{la.get('broken_markers')} | {samp[:120]}… |"
    )

lines += [
    "",
    "## Cross-cutting findings",
    "",
    "1. **Not empty collapse:** answers are English CoT; flexible-extract finds digits; accuracy still ~0–7%.",
    "2. **Uniform greedy disabled in sampler:** argmax(q_xs) would lock prior noise; forces ancestral — many warns in logs.",
    "3. **Val/ppl near C0:** conversion fits; Instruct generative math does not transfer under uniform reverse.",
    "4. **IFE ~13–18%** while GSM dies — short instruction following ≠ multi-step math.",
    "5. **Protocol asymmetry:** C0 DualCache/hubmatch thr=1 greedy vs uniform baseline ancestral — real, but word-salad algebra is still a uniform generative failure mode.",
    "6. **hybrid_p10 `1836811`:** never got Instruct lm-eval.",
    "7. **Systematic across recipes:** baseline, shift, anneal, continue-FT, xfer, joint U2 — same pattern ⇒ family issue, not one bad seed.",
    "",
    "## Falsifiers / next probes",
    "",
    "1. C0 GSM with `DECODE_PROFILE=baseline` ancestral (fair decode match).",
    "2. U0 NFE×GSM steps 32/64/128 + `log_samples`.",
    "3. 20-prompt sheet: C0 dual vs U0 ancestral.",
    "4. Instruct eval for hybrid `1836811`.",
    "5. Cite GenPPL+H separately from GSM — they disagree on purpose.",
    "",
]

out_md = DOC / "AUDIT_UNIFORM_HYBRID_GSM_COLLAPSE_2026-09-20.md"
out_json = DOC / "AUDIT_UNIFORM_HYBRID_GSM_COLLAPSE_2026-09-20.json"
out_md.write_text("\n".join(lines) + "\n")
out_json.write_text(json.dumps(rows, indent=2, default=str))
print(f"wrote {out_md}")
print(f"wrote {out_json}")
print("\n=== SUMMARY ===")
for r in rows:
    sc = r.get("scores") or {}
    gp = r.get("gen_ppl") or {}
    val = (r.get("val") or {}).get("val/ppl")
    la = r.get("log_audit") or {}
    print(
        f"{r['job']} {r['label'][:24]:24} val={fmt(val,2)} gsm={fmt((sc.get('gsm') or {}).get('score'))} "
        f"ife={fmt((sc.get('ife') or {}).get('score'))} gen={fmt(gp.get('ppl'),1)} "
        f"H={fmt(gp.get('H'),2)} warns={la.get('greedy_uniform_warns')}"
    )
