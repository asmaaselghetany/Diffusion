#!/usr/bin/env python3
"""Upload presentation-axis eval scores to WandB (entity aselghetany-nu).

Creates/updates one run per cell under project ``thesis_axes`` with summary
metrics (GSM / IFEval / MMLU / GenPPL) and axis tags. Does not embed secrets.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Scores from docs/research/PRESENTATION_SNAPSHOT_2026-09-18.md (audited).
# None = queued / not yet citeable.
AXES: List[Dict[str, Any]] = [
    # Paradigm: Native
    dict(
        run_id="axis_native_scratch_1836796",
        name="Native · scratch block (masked)",
        paradigm="native",
        corruption="masked",
        components="none",
        geometry="fixed-32",
        job_id="1836796",
        gsm=0.5,
        ife=8.1,
        mmlu=23.0,
        gen_ppl=None,
        status="ready",
    ),
    # Paradigm: AR → block
    dict(
        run_id="axis_ar2block_baseline_masked_1762534",
        name="AR→block · baseline (masked)",
        paradigm="ar2block",
        corruption="masked",
        components="none",
        geometry="fixed-32",
        job_id="1762534",
        gsm=62.1,
        ife=20.5,
        mmlu=39.8,
        gen_ppl=128.0,
        gen_ppl_H=4.80,
        status="ready",
    ),
    dict(
        run_id="axis_ar2block_math_mix_1763301",
        name="AR→block · math mix (masked)",
        paradigm="ar2block",
        corruption="masked",
        components="shift+comp+mask_sched+plain_ce+hub_parity",
        geometry="fixed-32",
        job_id="1763301",
        gsm=66.8,
        ife=22.4,
        mmlu=30.6,
        gen_ppl=89.0,
        gen_ppl_H=4.76,
        status="ready",
    ),
    dict(
        run_id="axis_ar2block_chat_mix_1773298",
        name="AR→block · chat mix (masked)",
        paradigm="ar2block",
        corruption="masked",
        components="shift+comp+mask_sched+plain_ce+hub_parity",
        geometry="fixed-32",
        job_id="1773298",
        gsm=43.4,
        ife=23.8,
        mmlu=40.3,
        gen_ppl=None,
        status="ready",
    ),
    dict(
        run_id="axis_ar2block_mixed_sft_1836800",
        name="AR→block · mixed SFT (masked)",
        paradigm="ar2block",
        corruption="masked",
        components="shift+comp+mask_sched+plain_ce+hub_parity",
        geometry="fixed-32",
        job_id="1836800",
        gsm=59.8,
        ife=23.3,
        mmlu=38.1,
        gen_ppl=None,
        status="ready",
    ),
    dict(
        run_id="axis_ar2block_baseline_uniform_1849335",
        name="AR→block · baseline (uniform)",
        paradigm="ar2block",
        corruption="uniform",
        components="none",
        geometry="fixed-32",
        job_id="1849335",
        gsm=None,
        ife=10.7,
        mmlu=None,
        gen_ppl=61.0,
        gen_ppl_H=6.04,
        status="partial",
    ),
    dict(
        run_id="axis_ar2block_bgmix_masked_1855537",
        name="AR→block · block-size mix (masked)",
        paradigm="ar2block",
        corruption="masked",
        components="none",
        geometry="weights_1_32",
        job_id="1855537",
        gsm=63.3,
        ife=20.7,
        mmlu=40.2,
        gen_ppl=114.0,
        status="ready",
    ),
    dict(
        run_id="axis_ar2block_bgmix_uniform_1849287",
        name="AR→block · block-size mix (uniform)",
        paradigm="ar2block",
        corruption="uniform",
        components="none",
        geometry="weights_1_32",
        job_id="1849287",
        gsm=None,
        ife=None,
        mmlu=None,
        gen_ppl=55.0,
        gen_ppl_H=6.25,
        status="queued",
    ),
    # Paradigm: AR → full sequence
    dict(
        run_id="axis_ar_sft_1857323",
        name="AR→full · matched AR SFT",
        paradigm="ar_full",
        corruption="n/a",
        components="none",
        geometry="n/a",
        job_id="1857323",
        gsm=None,
        ife=None,
        mmlu=None,
        gen_ppl=None,
        status="queued",
    ),
    # Paradigm: Joint
    dict(
        run_id="axis_joint_masked_1836723",
        name="Joint · AR+causal_clean (masked)",
        paradigm="joint",
        corruption="masked",
        components="joint_ar+causal_clean",
        geometry="fixed-32",
        job_id="1836723",
        gsm=38.5,
        ife=25.0,
        mmlu=42.0,
        gen_ppl=None,
        status="ready",
    ),
    dict(
        run_id="axis_joint_uniform_1848844",
        name="Joint · AR+causal_clean (uniform)",
        paradigm="joint",
        corruption="uniform",
        components="joint_ar+causal_clean",
        geometry="fixed-32",
        job_id="1848844",
        gsm=None,
        ife=None,
        mmlu=None,
        gen_ppl=87.0,
        status="queued",
    ),
    dict(
        run_id="axis_joint_all5_1836725",
        name="Joint · all five + joint (masked)",
        paradigm="joint",
        corruption="masked",
        components="shift+comp+mask_sched+plain_ce+hub_parity+joint_ar+causal_clean",
        geometry="fixed-32",
        job_id="1836725",
        gsm=44.0,
        ife=21.1,
        mmlu=29.8,
        gen_ppl=None,
        status="ready",
    ),
    # Components ablations (AR→block masked)
    dict(
        run_id="axis_ablate_shift_1856101",
        name="Ablate · +shift",
        paradigm="ar2block",
        corruption="masked",
        components="shift",
        geometry="fixed-32",
        job_id="1856101",
        gsm=64.0,
        ife=21.1,
        mmlu=33.7,
        gen_ppl=88.0,
        status="ready",
    ),
    dict(
        run_id="axis_ablate_comp_1856103",
        name="Ablate · +complementary",
        paradigm="ar2block",
        corruption="masked",
        components="complementary",
        geometry="fixed-32",
        job_id="1856103",
        gsm=53.3,
        ife=18.1,
        mmlu=37.3,
        gen_ppl=108.0,
        status="ready",
    ),
    dict(
        run_id="axis_ablate_shift_comp_1856105",
        name="Ablate · +shift+complementary",
        paradigm="ar2block",
        corruption="masked",
        components="shift+complementary",
        geometry="fixed-32",
        job_id="1856105",
        gsm=50.8,
        ife=17.0,
        mmlu=32.0,
        gen_ppl=107.0,
        status="ready",
    ),
    dict(
        run_id="axis_ablate_anneal_1857473",
        name="Ablate · +intra-block anneal",
        paradigm="ar2block",
        corruption="masked",
        components="intra_block_anneal",
        geometry="fixed-32",
        job_id="1857473",
        gsm=61.1,
        ife=21.4,
        mmlu=42.7,
        gen_ppl=134.0,
        status="ready",
    ),
    # Cross-corruption / hybrid
    dict(
        run_id="axis_u0_shift_1857278",
        name="Uniform · +shift",
        paradigm="ar2block",
        corruption="uniform",
        components="shift",
        geometry="fixed-32",
        job_id="1857278",
        gsm=None,
        ife=None,
        mmlu=None,
        gen_ppl=None,
        status="queued",
    ),
    dict(
        run_id="axis_u0_anneal_1857471",
        name="Uniform · +intra-block anneal",
        paradigm="ar2block",
        corruption="uniform",
        components="intra_block_anneal",
        geometry="fixed-32",
        job_id="1857471",
        gsm=None,
        ife=None,
        mmlu=None,
        gen_ppl=None,
        status="queued",
    ),
    dict(
        run_id="axis_hybrid_joint_curr_1857475",
        name="Hybrid · anneals (joint curriculum)",
        paradigm="ar2block",
        corruption="hybrid",
        components="intra_block_anneal+kernel_anneal",
        geometry="fixed-32",
        job_id="1857475",
        gsm=None,
        ife=None,
        mmlu=None,
        gen_ppl=106.0,
        status="queued",
    ),
    dict(
        run_id="axis_u0_from_c0_1857519",
        name="Uniform · continue-FT from masked baseline",
        paradigm="ar2block",
        corruption="uniform",
        components="none",
        geometry="fixed-32",
        job_id="1857519",
        gsm=None,
        ife=None,
        mmlu=None,
        gen_ppl=None,
        status="queued",
    ),
    dict(
        run_id="axis_hybrid_p10_1836811",
        name="Hybrid · p(uniform)=0.1",
        paradigm="ar2block",
        corruption="hybrid",
        components="none",
        geometry="fixed-32",
        job_id="1836811",
        gsm=None,
        ife=None,
        mmlu=None,
        gen_ppl=None,
        status="queued",
    ),
]


def _load_key() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    candidates = [
        os.path.expanduser("~/.config/wandb/api_key"),
    ]
    env_file = os.environ.get("WANDB_API_KEY_FILE", "").strip()
    if env_file:
        candidates.insert(0, os.path.expanduser(env_file))
    for path in candidates:
        if os.path.isfile(path):
            os.environ["WANDB_API_KEY"] = open(path).read().strip()
            return
    raise SystemExit(
        "WANDB_API_KEY missing; export it or set WANDB_API_KEY_FILE / "
        "~/.config/wandb/api_key")


def _metric_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status_code": {"ready": 1, "partial": 0.5, "queued": 0}.get(row["status"], -1)}
    for k in ("gsm", "ife", "mmlu", "gen_ppl", "gen_ppl_H"):
        v = row.get(k)
        if v is not None:
            out[k] = float(v)
    return out


def main() -> int:
    _load_key()
    import wandb

    entity = os.environ.get("WANDB_ENTITY", "aselghetany-nu")
    project = os.environ.get("WANDB_PROJECT", "thesis_axes")
    urls: List[str] = []
    for row in AXES:
        run = wandb.init(
            entity=entity,
            project=project,
            id=row["run_id"],
            name=row["name"],
            resume="allow",
            job_type="axis_scoreboard",
            tags=[
                f"paradigm:{row['paradigm']}",
                f"corruption:{row['corruption']}",
                f"geometry:{row['geometry']}",
                f"status:{row['status']}",
                f"job:{row['job_id']}",
            ],
            config={
                "paradigm": row["paradigm"],
                "corruption": row["corruption"],
                "components": row["components"],
                "geometry": row["geometry"],
                "job_id": row["job_id"],
                "status": row["status"],
                "source": "PRESENTATION_SNAPSHOT_2026-09-18",
            },
            settings=wandb.Settings(console="off", _disable_stats=True),
        )
        run.log(_metric_payload(row))
        for k, v in _metric_payload(row).items():
            run.summary[k] = v
        run.summary["components"] = row["components"]
        url = run.get_url()
        urls.append(url)
        print(f"OK {row['run_id']} -> {url}")
        run.finish()
    print(f"\nUploaded {len(urls)} axis runs to {entity}/{project}")
    print(f"Board: https://wandb.ai/{entity}/{project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
