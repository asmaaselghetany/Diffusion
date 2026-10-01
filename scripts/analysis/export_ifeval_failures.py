#!/usr/bin/env python3
"""Export IFEval failures for editable-token Phase-1 taxonomy.

Preferred input: lm-eval ``--log_samples`` JSONL under an ifeval output dir.
Fallback: parse ``lm_eval.log`` (prompts truncated + answers capped ~500 chars)
— **not** valid for the frozen go-threshold; use only for smoke.

Examples:
  # After submit_ifeval_samples.sh C0 finishes:
  .venv/bin/python scripts/analysis/export_ifeval_failures.py \\
    --model C0 --job 1762534 \\
    --samples-dir outputs/block_qwen/ar2block_masked_1762534/lm_eval_ifeval_samples_hubmatch/ifeval \\
    --out docs/research/editable_tokens

  .venv/bin/python scripts/analysis/export_ifeval_failures.py \\
    --model U0 --job 1955203 \\
    --samples-dir outputs/block_qwen/ar2block_uniform_1955203/lm_eval_ifeval_samples_uniform_commit/ifeval \\
    --out docs/research/editable_tokens
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from lm_eval.tasks.ifeval.utils import (
    InputExample,
    process_results,
    test_instruction_following_strict,
)

BLOCK_RE = re.compile(
    r"====================\nquestion: (.*?)\nanswer: (.*?)\n====================",
    re.S,
)


def extract_user_prompt(question: str) -> str | None:
    if "<|im_start|>user" in question:
        u = question.split("<|im_start|>user", 1)[1]
        return u.split("<|im_end|>", 1)[0].strip()
    return question.strip() or None


def match_gold_prompt(trunc: str, gold_by_prompt: dict[str, dict]) -> str | None:
    if trunc in gold_by_prompt:
        return trunc
    prefs = [p for p in gold_by_prompt if p.startswith(trunc)]
    if len(prefs) == 1:
        return prefs[0]
    for L in range(min(len(trunc), 200), 19, -1):
        prefs = [p for p in gold_by_prompt if p.startswith(trunc[:L])]
        if len(prefs) == 1:
            return prefs[0]
        if not prefs:
            break
    return None


def load_responses_from_samples_dir(samples_dir: Path) -> dict[str, str]:
    """Return prompt -> response from lm-eval samples_*.jsonl files."""
    paths = sorted(samples_dir.rglob("samples_*.jsonl"))
    if not paths:
        # some harness versions write .json
        paths = sorted(samples_dir.rglob("samples_*.json"))
    if not paths:
        raise SystemExit(f"No samples_*.jsonl under {samples_dir}")

    responses: dict[str, str] = {}
    for path in paths:
        if path.suffix == ".jsonl":
            lines = path.read_text(errors="replace").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt = obj.get("doc", {}).get("prompt") or obj.get("prompt")
                # lm-eval sample fields vary by version
                resp = None
                if "resps" in obj and obj["resps"]:
                    r0 = obj["resps"][0]
                    resp = r0[0] if isinstance(r0, list) else r0
                elif "filtered_resps" in obj and obj["filtered_resps"]:
                    resp = obj["filtered_resps"][0]
                elif "target" in obj and isinstance(obj.get("arguments"), list):
                    pass
                if prompt and resp is not None:
                    responses.setdefault(str(prompt), str(resp))
        else:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                for obj in data:
                    prompt = obj.get("doc", {}).get("prompt") or obj.get("prompt")
                    if "resps" in obj and obj["resps"]:
                        r0 = obj["resps"][0]
                        resp = r0[0] if isinstance(r0, list) else r0
                        if prompt:
                            responses.setdefault(str(prompt), str(resp))
    return responses


def load_responses_from_log(log_path: Path, gold_by_prompt: dict[str, dict]) -> dict[str, str]:
    text = log_path.read_text(errors="replace")
    marker = "=== task=ifeval ==="
    idx = text.find(marker)
    if idx < 0:
        idx = text.find("Selected Tasks: ['ifeval']")
    if idx < 0:
        raise SystemExit(f"No ifeval section found in {log_path}")
    after = text[idx:]
    m = re.search(r"\n=== task=(?!ifeval)", after)
    if m:
        after = after[: m.start()]

    responses: dict[str, str] = {}
    for q, a in BLOCK_RE.findall(after):
        trunc = extract_user_prompt(q)
        if not trunc:
            continue
        gold = match_gold_prompt(trunc, gold_by_prompt)
        if not gold:
            continue
        responses.setdefault(gold, a.strip())
    return responses


def bootstrap_ci(bits: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if not bits:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(bits)
    mean = sum(bits) / n
    means = []
    for _ in range(n_boot):
        s = sum(bits[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return mean, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--samples-dir", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilot-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.samples_dir and not args.log:
        raise SystemExit("Provide --samples-dir (preferred) or --log (truncated fallback)")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.job}"

    ds = load_dataset("google/IFEval", split="train")
    gold_by_prompt = {row["prompt"]: row for row in ds}

    source = "samples"
    if args.samples_dir:
        responses = load_responses_from_samples_dir(args.samples_dir)
    else:
        source = "log_truncated"
        responses = load_responses_from_log(args.log, gold_by_prompt)

    scored = []
    missing = 0
    for doc in ds:
        prompt = doc["prompt"]
        resp = responses.get(prompt)
        if resp is None:
            missing += 1
            continue
        metrics = process_results(doc, [resp])
        inp = InputExample(
            key=doc["key"],
            instruction_id_list=doc["instruction_id_list"],
            prompt=doc["prompt"],
            kwargs=doc["kwargs"],
        )
        strict_out = test_instruction_following_strict(inp, resp)
        scored.append(
            {
                "model": args.model,
                "job": args.job,
                "key": doc["key"],
                "prompt": prompt,
                "response": resp,
                "instruction_id_list": list(doc["instruction_id_list"]),
                "kwargs": doc["kwargs"],
                "prompt_level_strict_acc": bool(metrics["prompt_level_strict_acc"]),
                "prompt_level_loose_acc": bool(metrics["prompt_level_loose_acc"]),
                "follow_instruction_list_strict": list(strict_out.follow_instruction_list),
            }
        )

    n = len(scored)
    strict_bits = [1 if r["prompt_level_strict_acc"] else 0 for r in scored]
    loose_bits = [1 if r["prompt_level_loose_acc"] else 0 for r in scored]
    strict_mean, strict_lo, strict_hi = bootstrap_ci(strict_bits, seed=args.seed)
    loose_mean, loose_lo, loose_hi = bootstrap_ci(loose_bits, seed=args.seed + 1)

    fails = [r for r in scored if not r["prompt_level_strict_acc"]]
    by_fam: dict[str, list] = {}
    for r in fails:
        failed_ids = [
            iid
            for iid, ok in zip(r["instruction_id_list"], r["follow_instruction_list_strict"])
            if not ok
        ]
        fam = failed_ids[0].split(":")[0] if failed_ids else "unknown"
        r["failed_instruction_ids"] = failed_ids
        r["primary_failed_family"] = fam
        by_fam.setdefault(fam, []).append(r)

    rng = random.Random(args.seed)
    pilot: list = []
    fams = sorted(by_fam.keys())
    idxs = {f: 0 for f in fams}
    for f in fams:
        rng.shuffle(by_fam[f])
    while len(pilot) < min(args.pilot_n, len(fails)):
        progressed = False
        for f in fams:
            i = idxs[f]
            if i < len(by_fam[f]) and len(pilot) < args.pilot_n:
                pilot.append(by_fam[f][i])
                idxs[f] = i + 1
                progressed = True
        if not progressed:
            break

    scored_path = out / f"{tag}_scored.jsonl"
    fails_path = out / f"{tag}_failures.jsonl"
    pilot_path = out / f"{tag}_pilot_label_sheet.csv"
    summary_path = out / f"{tag}_summary.json"

    with scored_path.open("w") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with fails_path.open("w") as f:
        for r in fails:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fieldnames = [
        "item_id",
        "model",
        "job",
        "key",
        "primary_failed_family",
        "failed_instruction_ids",
        "instruction_id_list",
        "prompt",
        "response",
        "primary_label",
        "mixed",
        "annotator",
        "notes",
    ]
    with pilot_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(pilot):
            w.writerow(
                {
                    "item_id": f"{tag}_{i:03d}",
                    "model": args.model,
                    "job": args.job,
                    "key": r["key"],
                    "primary_failed_family": r["primary_failed_family"],
                    "failed_instruction_ids": "|".join(r["failed_instruction_ids"]),
                    "instruction_id_list": "|".join(r["instruction_id_list"]),
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "primary_label": "",
                    "mixed": "",
                    "annotator": "",
                    "notes": "",
                }
            )

    summary = {
        "model": args.model,
        "job": args.job,
        "source": source,
        "valid_for_go_threshold": source == "samples",
        "n_dataset": len(ds),
        "n_matched": n,
        "n_missing_prompts": missing,
        "prompt_level_strict_acc": {
            "mean": strict_mean,
            "ci95": [strict_lo, strict_hi],
            "n": n,
        },
        "prompt_level_loose_acc": {
            "mean": loose_mean,
            "ci95": [loose_lo, loose_hi],
            "n": n,
        },
        "n_failures_strict": len(fails),
        "failure_family_counts": dict(Counter(r["primary_failed_family"] for r in fails)),
        "pilot_n": len(pilot),
        "artifacts": {
            "scored": str(scored_path),
            "failures": str(fails_path),
            "pilot_sheet": str(pilot_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
