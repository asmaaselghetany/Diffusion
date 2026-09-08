# Experiment: Instruct collapse control

## Design

Separate **sampler/decode** vs **block training/recipe** as the source of free-gen soup.

| Arm | What |
|-----|------|
| **A** | HF `Qwen2.5-1.5B-Instruct` AR `generate` on instruction prompts |
| **B** | `BlockSampler` on **fresh AR-init** BlockTrainer weights (no block SFT) for `block_masked` and `block_uniform` |

Intended decision rule:

- If A fluent and B collapsed → sampler/decode path.
- If A fluent and B fluent → collapse from block training/recipe.
- If A bad → env/tokenizer issue.

## Jobs

| Role | Job ID | State | Elapsed |
|------|--------|-------|---------|
| Control | 1491190 | COMPLETED | 00:03:45 |

## Artifacts

- Output dir: `experiments/2026-08-25_collapse_control/artifacts/`
- `control_report.json`, `ar_instruct_samples.txt`, `block_ar_init_*_samples.txt`
- Tool: `tools/run_instruct_collapse_control.py`
- Submit: `sbatch scripts/slurm/instruct_collapse_control.sbatch`

## Diagnostics

### A — AR Instruct

Fluent English on all 4 prompts (Rayleigh scattering, photosynthesis, breakfast ideas). `n_collapsed=0` (correct).

### B — Block sampler on AR-init (no block SFT)

**Human reading: soup / multilingual gibberish** for both `block_masked` and `block_uniform` (L=256, bs=32, steps=32). Examples:

- masked: `gün_item-girl_msg怎么办 poised..header or.-generation also#if줄{/热心_COUNTER...`
- uniform: `伺"$"$"$"$"$инд max__ Desktop Desktop B A NotFoundException...`

Auto-heuristic reported `n_collapsed=0` because it only flags low uniqueness / high top-token mass / char loops — **high-entropy gibberish slips through as “fluent”.** Auto verdict in `control_report.json` is therefore **wrong**.

## Problems

1. Collapse detector misses high-entropy multilingual soup (needs readability / language prior, not just uniqueness).
2. Arm B is free-gen from AR-init under block attention — soup here implicates **decode/sampler + AR→block attention mismatch**, not only post-SFT recipe.

## Verdict

**Human override:** AR Instruct fluent; block-on-AR-init is soup → **sampler / block-decode path (and AR weights under block attention without SFT) are implicated**, before blaming Nemotron SFT alone. Auto-report string claiming “block training/recipe” must be ignored. Next: fix heuristic; keep D4 / ARPC+mixture as recipe interventions that may still be needed on top of a working decode path.
