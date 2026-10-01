# Research notes (optional)

Working notes for the Qwen **four-arm** bakeoff (init × corruption).  
Not part of the upstream UNI-D² docs layout.

| Doc | Role |
|-----|------|
| [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) | **Thesis experiments:** AR→block design space (A–E) + longitudinal eval bundles |
| [DESIGN_LOCKS.md](DESIGN_LOCKS.md) | Closed design decisions (B4v1, DualCache, C5 streams, tok/s) |
| [FOUR_ARMS_DIRECTIONS.md](FOUR_ARMS_DIRECTIONS.md) | Status of the four clean-graph arms + follow-on directions |
| [BLOCK_QWEN_TRAINING.md](BLOCK_QWEN_TRAINING.md) | Cluster launch / eval details |
| [LEVERS.md](LEVERS.md) | Lever registry (hooks off by default) |
| [BLOCKGEN_LEVERS.md](BLOCKGEN_LEVERS.md) | BlockGen-derived mixture / ARPC notes |
| [BASELINE.md](BASELINE.md) | Masked vs uniform baseline framing |
| [BAKEOFF_2026-09-24.md](BAKEOFF_2026-09-24.md) | **Phase-0 bake scoreboard** (C0/U0 remask; as of 2026-09-30) |
| [AR2BLOCK_GSM_TABLE.md](AR2BLOCK_GSM_TABLE.md) | **BlockGen-style GSM table** + `AR2B-U-floor` |
| [HARVEST_ATTACK_C3_2026-09-27.md](HARVEST_ATTACK_C3_2026-09-27.md) | Attack harvest + C3 |
| [C3_UNIFUSION_GAP_2026-09-30.md](C3_UNIFUSION_GAP_2026-09-30.md) | **Why C3 v1 failed vs Unifusion paper** |
| [UCC_VS_MASKED_CONF_2026-09-25.md](UCC_VS_MASKED_CONF_2026-09-25.md) | UCC sticky_min vs masked force-max forensic |
| [UNIFORM_CONFIDENCE_COMMIT_2026-09-24.md](UNIFORM_CONFIDENCE_COMMIT_2026-09-24.md) | UCC method note (homemade; not DualCache) |
| [FAIR_AR2BLOCK_MAP.md](FAIR_AR2BLOCK_MAP.md) | Fair both-arm design map |
| [THESIS_EXPERIMENT_PROTOCOL_2026-09-27.md](THESIS_EXPERIMENT_PROTOCOL_2026-09-27.md) | Preregistered floor / Layer A–C rules |
| [RESULTS_LEDGER_2026-09-20.md](RESULTS_LEDGER_2026-09-20.md) | Older axis harvest (pre-bake; see BAKEOFF for Sep 24+) |
| [VERIFICATION.md](VERIFICATION.md) | Verification tiers |
| [RESEARCH_LOGIC.md](RESEARCH_LOGIC.md) | Longer research narrative |

Canonical product docs for this fork addition: [../BLOCK_PATH.md](../BLOCK_PATH.md) and [../examples/block_qwen.md](../examples/block_qwen.md).
