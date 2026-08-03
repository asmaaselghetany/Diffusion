#!/usr/bin/env python3
"""Render block-diffusion architecture diagram for presentations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, xy, w, h, text, face, edge="#2E4A66", fontsize=9, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1B2430",
        weight=weight,
        zorder=3,
        linespacing=1.25,
    )
    return patch


def _arrow(ax, start, end, color="#5D7A9A"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color,
        zorder=1,
    )
    ax.add_patch(arr)


def render(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    fig.patch.set_facecolor("#F5F7FA")
    ax.set_facecolor("#F5F7FA")
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.15, 6.5)
    ax.axis("off")

    c_hub = "#DDE8F4"
    c_backbone = "#C8DBF0"
    c_trainer = "#B8D4C8"
    c_masked = "#F5D9B8"
    c_uniform = "#E8D4F0"
    c_loss = "#F0E6B8"
    c_sample = "#D4E8F0"

    _box(ax, (4.2, 5.55), 3.6, 0.7, "AR Qwen2.5 weights (HF hub)", c_hub, fontsize=10, bold=True)
    _arrow(ax, (6.0, 5.55), (6.0, 5.05))

    _box(
        ax,
        (2.4, 4.15),
        7.2,
        0.9,
        "QwenBlockForCausalLM\nconcat(xₜ, x₀)  ·  block_diff attention mask",
        c_backbone,
        fontsize=10,
        bold=True,
    )
    _arrow(ax, (6.0, 4.15), (6.0, 3.75))

    _box(
        ax,
        (3.6, 2.95),
        4.8,
        0.8,
        "BlockTrainer  (Lightning · Hydra config)",
        c_trainer,
        fontsize=10,
        bold=True,
    )

    _arrow(ax, (4.8, 2.95), (2.4, 2.45))
    _arrow(ax, (7.2, 2.95), (9.6, 2.45))

    _box(
        ax,
        (0.35, 1.35),
        4.0,
        1.1,
        "Masked arm\nBlockMaskedForward · per-block t\nSUBS ELBO",
        c_masked,
        fontsize=9,
    )
    _box(
        ax,
        (7.65, 1.35),
        4.0,
        1.1,
        "Uniform arm (BlockGen)\nBlockUniformForward · π = 1/V\nDUO_BASE ELBO",
        c_uniform,
        fontsize=9,
    )

    ax.text(
        6.0,
        2.55,
        "forward_process_name",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#5D7A9A",
        style="italic",
    )

    _arrow(ax, (2.35, 1.35), (4.5, 0.95))
    _arrow(ax, (9.65, 1.35), (7.5, 0.95))

    _box(ax, (4.0, 0.35), 4.0, 0.6, "ELBO loss  (losses/block_elbo.py)", c_loss, fontsize=9, bold=True)
    _arrow(ax, (6.0, 0.35), (6.0, 0.05))

    _box(
        ax,
        (2.8, -0.05),
        6.4,
        0.55,
        "BlockSampler (G6): semi-AR blocks · unmask (masked) | redraw (uniform)",
        c_sample,
        fontsize=9,
    )

    ax.text(
        0.35,
        6.15,
        "One backbone · one trainer · config switches the forward process",
        fontsize=10,
        color="#1B2430",
        weight="bold",
    )
    ax.text(
        0.35,
        5.85,
        "Not on this path: BD3LM / BlockDiT",
        fontsize=8.5,
        color="#5D7A9A",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/tmp/block_architecture.png"),
    )
    args = parser.parse_args()
    render(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
