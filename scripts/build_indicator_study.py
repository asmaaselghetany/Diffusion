#!/usr/bin/env python
"""Build a brief collaborator-facing P0-P4 study bundle from gate artifacts.

This script aggregates:
  - P0 evidence from pytest and sampler-smoke logs
  - P1..P4 evidence from an indicator_gate.json artifact

Outputs:
  - study_summary.md
  - indicator_table.csv
  - indicator_assessment.json
  - fig_p0_p4_status.png
  - fig_key_metrics.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Check:
    priority: str
    criterion: str
    status: bool
    value: str
    source: str


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(_load_text(path))


def _parse_pytest_all(pytest_all_log: str) -> tuple[bool, str]:
    passed_match = re.search(r"(\d+)\s+passed", pytest_all_log)
    failed_match = re.search(r"(\d+)\s+failed", pytest_all_log)
    errors_match = re.search(r"(\d+)\s+error", pytest_all_log)

    n_passed = int(passed_match.group(1)) if passed_match else 0
    n_failed = int(failed_match.group(1)) if failed_match else 0
    n_errors = int(errors_match.group(1)) if errors_match else 0
    ok = (n_passed > 0) and (n_failed == 0) and (n_errors == 0)
    return ok, f"passed={n_passed}, failed={n_failed}, errors={n_errors}"


def _extract_test_section(pytest_regression_log: str, test_file: str) -> str:
    pattern = re.compile(
        rf"=== RUN {re.escape(test_file)} ===(?P<body>.*?)(?=\n=== RUN |\n=== COMPLETE ===|\Z)",
        re.DOTALL,
    )
    match = pattern.search(pytest_regression_log)
    return match.group("body") if match else ""


def _section_passed(section: str) -> tuple[bool, str]:
    if not section:
        return False, "section-missing"
    passed_match = re.search(r"(\d+)\s+passed", section)
    failed_match = re.search(r"(\d+)\s+failed", section)
    n_passed = int(passed_match.group(1)) if passed_match else 0
    n_failed = int(failed_match.group(1)) if failed_match else 0
    ok = (n_passed > 0) and (n_failed == 0)
    return ok, f"passed={n_passed}, failed={n_failed}"


def _fmt_float(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(x):
        return f"{x:.{ndigits}f}"
    return str(x)


def _build_checks(
    indicator: dict[str, Any],
    pytest_all_text: str,
    pytest_reg_text: str,
    smoke_text: str,
    indicator_json_path: Path,
    pytest_all_path: Path,
    pytest_reg_path: Path,
    smoke_path: Path,
) -> list[Check]:
    checks: list[Check] = []

    # P0
    p0_unit_ok, p0_unit_value = _parse_pytest_all(pytest_all_text)
    checks.append(
        Check(
            priority="P0",
            criterion="All unit tests pass on CPU",
            status=p0_unit_ok,
            value=p0_unit_value,
            source=str(pytest_all_path),
        )
    )

    oracle_section = _extract_test_section(pytest_reg_text, "tests/test_oracle_reverse.py")
    oracle_ok, oracle_value = _section_passed(oracle_section)
    checks.append(
        Check(
            priority="P0",
            criterion="Oracle DDIM reconstructs",
            status=oracle_ok,
            value=oracle_value,
            source=str(pytest_reg_path),
        )
    )

    diffusers_section = _extract_test_section(pytest_reg_text, "tests/test_against_diffusers.py")
    diff_ok, diff_value = _section_passed(diffusers_section)
    checks.append(
        Check(
            priority="P0",
            criterion="Reference comparison matches tolerance",
            status=diff_ok,
            value=diff_value,
            source=str(pytest_reg_path),
        )
    )

    smoke_ok = ("ok=True" in smoke_text) and ("sampler smoke passed" in smoke_text)
    checks.append(
        Check(
            priority="P0",
            criterion="No NaNs in sampler smoke tests",
            status=smoke_ok,
            value=smoke_text.strip().replace("\n", "; "),
            source=str(smoke_path),
        )
    )

    # P1
    p1_tok = bool(indicator.get("p1_token_pipeline", {}).get("ok", False))
    p1_legacy = bool(indicator.get("p1_legacy_flag", {}).get("ok", False))
    checks.append(
        Check(
            priority="P1",
            criterion="Token-ID embedding pipeline end-to-end",
            status=p1_tok,
            value=f"ok={p1_tok}",
            source=str(indicator_json_path),
        )
    )
    checks.append(
        Check(
            priority="P1",
            criterion="Legacy contextual pipeline still runs under flag",
            status=p1_legacy,
            value=f"ok={p1_legacy}",
            source=str(indicator_json_path),
        )
    )

    # P1bis
    fm = indicator.get("p1bis_flow_matching", {})
    imf = indicator.get("p1bis_imf", {})
    final_block = indicator.get("final_genppl", {})

    fm_ok = bool(fm.get("ok", False))
    imf_ok = bool(imf.get("ok", False))
    jvp_ok = bool(imf.get("jvp_probe_ok", False))
    alpha_start = float(imf.get("alpha_start", 0.0))
    alpha_end = float(imf.get("alpha_end", 0.0))
    curriculum_ok = alpha_end > alpha_start
    proxy_gen_ok = bool(final_block.get("proxy_pass", False))
    if "proxy_pass" not in final_block:
        proxy_gen_ok = bool(final_block.get("one_step_pass", False) and final_block.get("few_step_pass", False))
    diversity_one = final_block.get("one_step_diversity", {})
    quality_pass = bool(final_block.get("quality_pass", False))

    checks.append(
        Check(
            priority="P1bis",
            criterion="Flow matching baseline trains and samples",
            status=fm_ok,
            value=f"loss_start={_fmt_float(fm.get('loss_start'))}, loss_end={_fmt_float(fm.get('loss_end'))}",
            source=str(indicator_json_path),
        )
    )
    checks.append(
        Check(
            priority="P1bis",
            criterion="iMF loss runs without JVP operator failures",
            status=bool(imf_ok and jvp_ok),
            value=f"imf_ok={imf_ok}, jvp_ok={jvp_ok}",
            source=str(indicator_json_path),
        )
    )
    checks.append(
        Check(
            priority="P1bis",
            criterion="One-step/few-step proxy generation meets GenPPL gate",
            status=proxy_gen_ok,
            value=(
                f"proxy_pass={proxy_gen_ok}, quality_pass={quality_pass}, "
                f"distinct_2={_fmt_float(diversity_one.get('distinct_2'))}"
            ),
            source=str(indicator_json_path),
        )
    )
    checks.append(
        Check(
            priority="P1bis",
            criterion="Curriculum knob active (alpha increases)",
            status=curriculum_ok,
            value=f"alpha_start={_fmt_float(alpha_start)}, alpha_end={_fmt_float(alpha_end)}",
            source=str(indicator_json_path),
        )
    )

    # P2
    p2 = indicator.get("p2_joint_ce", {})
    p2_ok = bool(p2.get("ok", False))
    stage2_disabled = not bool(p2.get("stage2_enabled", True))
    checks.append(
        Check(
            priority="P2",
            criterion="CE head trained jointly in stage-1",
            status=p2_ok,
            value=f"ce_loss={_fmt_float(p2.get('ce_loss'))}",
            source=str(indicator_json_path),
        )
    )
    checks.append(
        Check(
            priority="P2",
            criterion="Stage-2 decoder not required by default",
            status=stage2_disabled,
            value=f"stage2_enabled={p2.get('stage2_enabled')}",
            source=str(indicator_json_path),
        )
    )

    # P3
    p3 = indicator.get("p3_stabilizers", {})
    p3_ok = bool(p3.get("ok", False))
    sc_ablation_ok = p3.get("self_cond_off_loss") is not None and p3.get("self_cond_on_loss") is not None
    balancing_ok = p3.get("ddpm_min_snr_loss") is not None
    checks.append(
        Check(
            priority="P3",
            criterion="Self-conditioning wired and ablated",
            status=bool(p3_ok and sc_ablation_ok),
            value=f"off={_fmt_float(p3.get('self_cond_off_loss'))}, on={_fmt_float(p3.get('self_cond_on_loss'))}",
            source=str(indicator_json_path),
        )
    )
    checks.append(
        Check(
            priority="P3",
            criterion="Timestep balancing path implemented (Min-SNR/DDPM)",
            status=bool(p3_ok and balancing_ok),
            value=f"ddpm_min_snr_loss={_fmt_float(p3.get('ddpm_min_snr_loss'))}",
            source=str(indicator_json_path),
        )
    )

    # P4
    p4 = indicator.get("p4_infill", {})
    p4_ok = bool(p4.get("ok", False))
    pre_acc = float(p4.get("pre_masked_token_acc", 0.0))
    post_acc = float(p4.get("post_masked_token_acc", 0.0))
    infill_improved = post_acc > pre_acc
    checks.append(
        Check(
            priority="P4",
            criterion="Span masking trained (not sampling-only)",
            status=p4_ok,
            value=f"pre={_fmt_float(pre_acc)}, post={_fmt_float(post_acc)}",
            source=str(indicator_json_path),
        )
    )
    checks.append(
        Check(
            priority="P4",
            criterion="Infill metrics move from ~0 to non-trivial",
            status=bool(p4_ok and infill_improved),
            value=f"delta={_fmt_float(post_acc - pre_acc)}",
            source=str(indicator_json_path),
        )
    )

    return checks


def _priority_rollup(checks: list[Check]) -> dict[str, bool]:
    priorities = ["P0", "P1", "P1bis", "P2", "P3", "P4"]
    out: dict[str, bool] = {}
    for p in priorities:
        own = [c.status for c in checks if c.priority == p]
        out[p] = bool(own) and all(own)
    return out


def _write_csv(checks: list[Check], out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["priority", "criterion", "status", "value", "source"],
        )
        writer.writeheader()
        for c in checks:
            writer.writerow(
                {
                    "priority": c.priority,
                    "criterion": c.criterion,
                    "status": "PASS" if c.status else "FAIL",
                    "value": c.value,
                    "source": c.source,
                }
            )


def _write_assessment_json(
    checks: list[Check],
    rollup: dict[str, bool],
    out_json: Path,
    indicator: dict[str, Any],
    provenance: dict[str, str],
) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rollup": {k: bool(v) for k, v in rollup.items()},
        "checks": [
            {
                "priority": c.priority,
                "criterion": c.criterion,
                "status": bool(c.status),
                "value": c.value,
                "source": c.source,
            }
            for c in checks
        ],
        "provenance": provenance,
        "raw_indicator": indicator,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_rollup(rollup: dict[str, bool], out_png: Path) -> None:
    labels = list(rollup.keys())
    values = [1.0 if rollup[k] else 0.0 for k in labels]
    colors = ["#2ca02c" if rollup[k] else "#d62728" for k in labels]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Pass (1) / Fail (0)")
    ax.set_title("P0-P4 Indicator Rollup")
    ax.grid(axis="y", alpha=0.25)

    for idx, v in enumerate(values):
        ax.text(idx, v + 0.03, "PASS" if v > 0.5 else "FAIL", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_key_metrics(indicator: dict[str, Any], out_png: Path) -> None:
    fm = indicator.get("p1bis_flow_matching", {})
    imf = indicator.get("p1bis_imf", {})
    p4 = indicator.get("p4_infill", {})
    final_block = indicator.get("final_genppl", {})

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))

    # FM losses
    ax = axes[0, 0]
    fm_vals = [float(fm.get("loss_start", float("nan"))), float(fm.get("loss_end", float("nan")))]
    ax.bar(["fm_start", "fm_end"], fm_vals, color=["#1f77b4", "#17becf"], edgecolor="black", linewidth=0.7)
    ax.set_title("Flow Matching Loss")
    ax.grid(axis="y", alpha=0.25)

    # iMF losses
    ax = axes[0, 1]
    imf_vals = [float(imf.get("loss_start", float("nan"))), float(imf.get("loss_end", float("nan")))]
    ax.bar(["imf_start", "imf_end"], imf_vals, color=["#ff7f0e", "#2ca02c"], edgecolor="black", linewidth=0.7)
    ax.set_title("iMF Loss")
    ax.grid(axis="y", alpha=0.25)

    # Infill pre/post
    ax = axes[1, 0]
    inf_vals = [
        float(p4.get("pre_masked_token_acc", float("nan"))),
        float(p4.get("post_masked_token_acc", float("nan"))),
    ]
    ax.bar(["pre", "post"], inf_vals, color=["#8c564b", "#9467bd"], edgecolor="black", linewidth=0.7)
    ax.set_title("Masked Infill Accuracy")
    ax.grid(axis="y", alpha=0.25)

    # Final GenPPL (context only)
    ax = axes[1, 1]
    one_ppl = float(final_block.get("one_step_ppl", float("nan")))
    few_ppl = float(final_block.get("few_step_ppl", float("nan")))
    ax.bar(["1-step", "few-step"], [one_ppl, few_ppl], color=["#e377c2", "#7f7f7f"], edgecolor="black", linewidth=0.7)
    ax.axhline(150.0, color="#d62728", linestyle="--", linewidth=1.0, label="1-step target")
    ax.axhline(100.0, color="#9467bd", linestyle="--", linewidth=1.0, label="few-step target")
    ax.set_title("External GenPPL (Context)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_markdown(
    checks: list[Check],
    rollup: dict[str, bool],
    indicator: dict[str, Any],
    provenance: dict[str, str],
    out_md: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Continuous Embedding P0-P4 Assessment")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- P0 rollup: {'PASS' if rollup.get('P0', False) else 'FAIL'}")
    lines.append(f"- P1 rollup: {'PASS' if rollup.get('P1', False) else 'FAIL'}")
    lines.append(f"- P1bis rollup: {'PASS' if rollup.get('P1bis', False) else 'FAIL'}")
    lines.append(f"- P2 rollup: {'PASS' if rollup.get('P2', False) else 'FAIL'}")
    lines.append(f"- P3 rollup: {'PASS' if rollup.get('P3', False) else 'FAIL'}")
    lines.append(f"- P4 rollup: {'PASS' if rollup.get('P4', False) else 'FAIL'}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Indicator JSON (P1-P4): `{provenance['indicator_json']}`")
    lines.append(f"- Pytest all log (P0 unit pass): `{provenance['pytest_all_log']}`")
    lines.append(f"- Pytest regression log (oracle/reference): `{provenance['pytest_regression_log']}`")
    lines.append(f"- Sampler smoke log (NaN guard): `{provenance['sampler_smoke_log']}`")
    lines.append("")
    lines.append("## Detailed Checklist")
    lines.append("")
    lines.append("| Priority | Criterion | Status | Value |")
    lines.append("|---|---|---|---|")
    for c in checks:
        status = "PASS" if c.status else "FAIL"
        lines.append(f"| {c.priority} | {c.criterion} | {status} | {c.value} |")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    fm = indicator.get("p1bis_flow_matching", {})
    imf = indicator.get("p1bis_imf", {})
    p4 = indicator.get("p4_infill", {})
    final_block = indicator.get("final_genppl", {})
    lines.append(
        f"- FM loss: start={_fmt_float(fm.get('loss_start'))}, end={_fmt_float(fm.get('loss_end'))}"
    )
    lines.append(
        f"- iMF loss: start={_fmt_float(imf.get('loss_start'))}, end={_fmt_float(imf.get('loss_end'))}, jvp_ok={imf.get('jvp_probe_ok')}"
    )
    lines.append(
        f"- Infill masked-token accuracy: pre={_fmt_float(p4.get('pre_masked_token_acc'))}, post={_fmt_float(p4.get('post_masked_token_acc'))}"
    )
    lines.append(
        f"- Final GenPPL context: one_step={_fmt_float(final_block.get('one_step_ppl'))}, few_step={_fmt_float(final_block.get('few_step_ppl'))}"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This report is focused on P0-P4 readiness indicators.")
    lines.append("- Final GenPPL is shown for context and remains a separate quality gate.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator-json", type=Path, required=True)
    parser.add_argument("--pytest-all-log", type=Path, required=True)
    parser.add_argument("--pytest-regression-log", type=Path, required=True)
    parser.add_argument("--sampler-smoke-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    indicator = _load_json(args.indicator_json)
    pytest_all_text = _load_text(args.pytest_all_log)
    pytest_reg_text = _load_text(args.pytest_regression_log)
    smoke_text = _load_text(args.sampler_smoke_log)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checks = _build_checks(
        indicator=indicator,
        pytest_all_text=pytest_all_text,
        pytest_reg_text=pytest_reg_text,
        smoke_text=smoke_text,
        indicator_json_path=args.indicator_json,
        pytest_all_path=args.pytest_all_log,
        pytest_reg_path=args.pytest_regression_log,
        smoke_path=args.sampler_smoke_log,
    )
    rollup = _priority_rollup(checks)
    provenance = {
        "indicator_json": str(args.indicator_json),
        "pytest_all_log": str(args.pytest_all_log),
        "pytest_regression_log": str(args.pytest_regression_log),
        "sampler_smoke_log": str(args.sampler_smoke_log),
    }

    csv_path = args.out_dir / "indicator_table.csv"
    md_path = args.out_dir / "study_summary.md"
    json_path = args.out_dir / "indicator_assessment.json"
    fig_rollup_path = args.out_dir / "fig_p0_p4_status.png"
    fig_metrics_path = args.out_dir / "fig_key_metrics.png"

    _write_csv(checks, csv_path)
    _write_markdown(checks, rollup, indicator, provenance, md_path)
    _write_assessment_json(checks, rollup, json_path, indicator, provenance)
    _plot_rollup(rollup, fig_rollup_path)
    _plot_key_metrics(indicator, fig_metrics_path)

    print(json.dumps({"rollup": rollup, "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
