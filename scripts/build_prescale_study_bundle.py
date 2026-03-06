#!/usr/bin/env python
"""Build an aggregated study bundle from pre-scale gate run folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = [
    "20260219_172216_gates_A_to_E",
    "20260219_172748_gates_A_to_E",
    "20260219_173249_gates_A_to_E",
    "20260219_173411_gates_A_to_E",
    "20260219_173744_gates_A_to_E",
]


def _to_row(summary: dict[str, Any], run_name: str, summary_path: Path) -> dict[str, Any]:
    gates = summary["gates"]
    gate_details = summary.get("gate_details", {})

    if "A" in gates:
        a_pass = bool(gates["A"]["pass"])
        a_pre_ce = gates["A"]["pre"]["ce_mean"] if gates["A"].get("pre") else float("nan")
        a_post_ce = gates["A"]["post"]["ce_mean"] if gates["A"].get("post") else float("nan")
        a_post_acc = gates["A"]["post"]["masked_acc_mean"] if gates["A"].get("post") else float("nan")
        a_gen_acc = gates["A"].get("gen_masked_acc_mean", float("nan"))
    else:
        a_overfit = bool(gates.get("A_overfit", {}).get("pass", False))
        a_recon = bool(gates.get("A_recon", {}).get("pass", False))
        a_pass = bool(a_overfit and a_recon)
        train_blob = gate_details.get("A_suite", {}).get("train", {})
        a_pre_ce = train_blob.get("pre", {}).get("ce_mean", float("nan"))
        a_post_ce = train_blob.get("post", {}).get("ce_mean", float("nan"))
        a_post_acc = train_blob.get("post", {}).get("masked_acc_mean", float("nan"))
        a_gen_acc = float("nan")

    if "D" in gates:
        d_pass = bool(gates["D"]["pass"])
        d_metrics = gates["D"]["fixed_dev_metrics"]
    else:
        d_stability = bool(gates.get("D_stability", {}).get("pass", False))
        d_signal = bool(gates.get("D_signal", {}).get("pass", False))
        d_pass = bool(d_stability and d_signal)
        d_blob = gate_details.get("D", {})
        d_metrics = d_blob.get("fixed_dev_metrics", {})

    return {
        "run": run_name,
        "objective": summary["runtime"]["objective"],
        "interpolant": summary["runtime"]["interpolant"],
        "device": summary["runtime"]["device"],
        "A_pass": a_pass,
        "A_overfit_pass": bool(gates.get("A_overfit", {}).get("pass", a_pass)),
        "A_recon_pass": bool(gates.get("A_recon", {}).get("pass", a_pass)),
        "A_train_pass": bool(gates.get("A_train", {}).get("pass", a_pass)),
        "B_pass": gates["B"]["pass"],
        "B_cuda_pass": bool(gates.get("B_cuda", {}).get("pass", True)),
        "C_pass": gates["C"]["pass"],
        "D_pass": d_pass,
        "D_stability_pass": bool(gates.get("D_stability", {}).get("pass", d_pass)),
        "D_signal_pass": bool(gates.get("D_signal", {}).get("pass", d_pass)),
        "F_pass": bool(gates.get("F", {}).get("pass", True)),
        "G_pass": bool(gates.get("G", {}).get("pass", True)),
        "E_pass": gates["E"]["pass"],
        "E2_pass": bool(gates.get("E2", {}).get("pass", True)),
        "A_pre_ce": a_pre_ce,
        "A_post_ce": a_post_ce,
        "A_post_masked_acc": a_post_acc,
        "A_gen_masked_acc": a_gen_acc,
        "D_masked_ce_mean": d_metrics.get("masked_ce_mean", float("nan")),
        "D_masked_acc_mean": d_metrics.get("masked_token_acc_mean", float("nan")),
        "E_nogc_tokens_per_sec": gates["E"]["profiles"][0]["tokens_per_sec"],
        "E_gc_tokens_per_sec": gates["E"]["profiles"][1]["tokens_per_sec"],
        "summary_json": str(summary_path),
        "summary_md": str(summary_path.with_name("summary.md")),
    }


def _write_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with (out_dir / "study_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (out_dir / "study_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _plot_gate_matrix(rows: list[dict[str, Any]], out_dir: Path) -> None:
    main_rows = [r for r in rows if "gate_b_cpu_ddp" not in str(r["run"])]
    if not main_rows:
        return
    labels = [str(r["run"]).replace("_gates_A_to_E", "") for r in main_rows]
    gate_keys = ["A_pass", "B_pass", "B_cuda_pass", "C_pass", "D_pass", "F_pass", "G_pass", "E_pass", "E2_pass"]
    mat = np.array([[1.0 if bool(r[k]) else 0.0 for k in gate_keys] for r in main_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3 + 0.5 * len(labels)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(gate_keys)), ["A", "B", "B_cuda", "C", "D", "F", "G", "E", "E2"])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Pre-scale Gate Matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "PASS" if mat[i, j] > 0.5 else "FAIL", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "gate_matrix.png", dpi=180)
    plt.close(fig)


def _plot_ce_acc(rows: list[dict[str, Any]], out_dir: Path) -> None:
    main_rows = [r for r in rows if "gate_b_cpu_ddp" not in str(r["run"])]
    if not main_rows:
        return
    labels = [str(r["run"]).replace("_gates_A_to_E", "") for r in main_rows]
    x = np.arange(len(labels))
    a_pre = [float(r["A_pre_ce"]) for r in main_rows]
    a_post = [float(r["A_post_ce"]) for r in main_rows]
    a_acc = [float(r["A_post_masked_acc"]) for r in main_rows]
    d_acc = [float(r["D_masked_acc_mean"]) for r in main_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x - 0.15, a_pre, width=0.3, label="Gate A pre CE", color="#7f7f7f")
    axes[0].bar(x + 0.15, a_post, width=0.3, label="Gate A post CE", color="#1f77b4")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_title("Gate A CE shift")
    axes[0].set_ylabel("Masked CE")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(x, a_acc, marker="o", label="Gate A post masked acc", color="#2ca02c")
    axes[1].plot(x, d_acc, marker="s", label="Gate D masked acc", color="#d62728")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylim(0, max(0.1, float(np.nanmax(np.array(a_acc + d_acc))) * 1.2))
    axes[1].set_title("Masked-token accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "ce_acc_comparison.png", dpi=180)
    plt.close(fig)


def _write_readme(rows: list[dict[str, Any]], out_dir: Path, cpu_gate_b_json: Path | None) -> None:
    lines = [
        "# Pre-Scale Gates Study Bundle",
        "",
        "## Included Runs",
        "",
    ]
    main_rows = [r for r in rows if "gate_b_cpu_ddp" not in str(r["run"])]
    for row in main_rows:
        lines.append(
            f"- `{row['run']}` objective=`{row['objective']}` gates: "
            f"A={row['A_pass']} B={row['B_pass']} B_cuda={row['B_cuda_pass']} C={row['C_pass']} D={row['D_pass']} "
            f"F={row['F_pass']} G={row['G_pass']} E={row['E_pass']} E2={row['E2_pass']}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `study_table.csv`",
            "- `study_table.json`",
            "- `gate_matrix.png`",
            "- `ce_acc_comparison.png`",
        ]
    )
    if cpu_gate_b_json is not None:
        lines.extend(
            [
                "",
                "## Dedicated Gate B (Multi-rank)",
                "",
                f"- `{cpu_gate_b_json}`",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=str,
        default="outputs/continuous_embedding/eval/prescale_gates",
        help="Directory containing run folders with summary.json files.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run folder name to include (can be repeated). Defaults to an internal run list.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/continuous_embedding/eval/prescale_gates/study_20260219",
        help="Output directory for aggregated study artifacts.",
    )
    parser.add_argument(
        "--cpu-gateb-json",
        type=str,
        default="outputs/continuous_embedding/eval/prescale_gates/20260219_172748_gates_A_to_E/gate_b_cpu_ddp/gate_b_jvp_ddp.json",
        help="Optional path to dedicated multi-rank Gate-B artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_names = args.run if args.run else DEFAULT_RUNS
    rows: list[dict[str, Any]] = []
    for run_name in run_names:
        summary_path = runs_root / run_name / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(_to_row(summary, run_name, summary_path))

    cpu_gate_b_json = Path(args.cpu_gateb_json)
    if cpu_gate_b_json.exists():
        gate_b = json.loads(cpu_gate_b_json.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": str(cpu_gate_b_json.parent.relative_to(runs_root)),
                "objective": "imf",
                "interpolant": "rectified",
                "device": "cpu",
                "A_pass": "n/a",
                "A_overfit_pass": "n/a",
                "A_recon_pass": "n/a",
                "A_train_pass": "n/a",
                "B_pass": gate_b["pass"],
                "B_cuda_pass": "n/a",
                "C_pass": "n/a",
                "D_pass": "n/a",
                "D_stability_pass": "n/a",
                "D_signal_pass": "n/a",
                "F_pass": "n/a",
                "G_pass": "n/a",
                "E_pass": "n/a",
                "E2_pass": "n/a",
                "A_pre_ce": "n/a",
                "A_post_ce": "n/a",
                "A_post_masked_acc": "n/a",
                "A_gen_masked_acc": "n/a",
                "D_masked_ce_mean": "n/a",
                "D_masked_acc_mean": "n/a",
                "E_nogc_tokens_per_sec": "n/a",
                "E_gc_tokens_per_sec": "n/a",
                "summary_json": str(cpu_gate_b_json),
                "summary_md": "n/a",
            }
        )

    if not rows:
        print("No summaries found; nothing written.")
        return 1

    _write_table(rows, out_dir)
    _plot_gate_matrix(rows, out_dir)
    _plot_ce_acc(rows, out_dir)
    _write_readme(rows, out_dir, cpu_gate_b_json if cpu_gate_b_json.exists() else None)

    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
