# BlockGen recipe notes (for our lever micros)

Clone: `third_party/blockgen` ([jdeschena/blockgen](https://github.com/jdeschena/blockgen), arXiv:2606.02241).
Read date: 2026-08-06. **Do not launch these levers until Track 1 1500 (`138143`/`138144`) returns.**

## What BlockGen actually does (uniform)

| Piece | BlockGen | Our `block_qwen` today |
|-------|----------|------------------------|
| Corruption | Uniform-state (Duo-style) via `BlockGenUniform` | `BlockUniformForwardProcess` + DUO/UDLM NLL — aligned in spirit |
| Default `loss_type` | **`elbo`** (`configs/algo/blockgen-uniform.yaml`) | `loss_type: elbo` — **already on**; not a lever micro |
| Special cases | `loss_type_special_cases` can force CE per block size; eval prefers ELBO for \(L'>1\) | We always ELBO on uniform; paper note: unweighted CE underperforms NELBO for uniform \(L'>4\) |
| Block sizes | Mixture via `block_weights` over **powers of 2** (`get_block_size` → `2**log_block_size`) | Fixed `block_size=32`; optional `algo.block_size_mixture` (uniform draw over list — **not** their weighted \(2^k\) / `u-stratified`) |
| Stratified block-size draw | `block_size_per_gpu: u-stratified` — stratifies **which block size** each GPU/accum step sees | **Not implemented.** Do not confuse with `algo.stratified_gamma` |
| Timestep / γ | Continuous `T: 0`; their “stratified” in code is the block-size path above | `algo.stratified_gamma` = **noise-level (t) stratification** within a fixed block — different axis |
| Scale | 170M DiT, 250k–1M steps, batch 512, LR 3e-4 | 1.5B Qwen, micros 500–1500, batch 128, LR 2e-5 |
| Decode | Ancestral + **ARPC** (needs mixture including block size 1) | `sampling.use_arpc`; keep off until mixture includes 1 |

### Name collision (do not conflate)

| Name in older notes | Actual axis | Ours / BlockGen |
|---------------------|-------------|-----------------|
| “stratified γ” (BlockGen-flavored) | **block-size** draw across GPUs (`u-stratified`) | BlockGen only; we have no knob |
| `algo.stratified_gamma` | **timestep / noise level** within block | Ours only |

## Mapping to our Hydra knobs (single-factor micros)

Script: `uni-d2/scripts/submit_blockgen_lever_micro.sh`

| LEVER | Override | Axis | Notes |
|-------|----------|------|-------|
| **A** | `algo.block_size_mixture=[16,32]` | BlockGen multi-size analogue | Closest shared recipe lever; random over list ≠ `u-stratified` |
| **B** | `algo.stratified_gamma=0.5` | Our **t**-strat | Distinct from A; was previously bundled into A by mistake |
| **C** | `algo.block_size_mixture=[1,32]` | AR-size component | Prerequisite for ARPC; still train-time mixture only |
| **D** | C + `sampling.use_arpc=true` | Decode | Only after C is understood |

**Not a lever:** `loss_type=elbo` — already default on both arms. Switching *to* CE would be an anti-BlockGen ablation, not a BlockGen-derived rescue.

**Do not copy blindly:** their `u-stratified` multi-GPU schedule, adaLN DiT backbone, TinyGSM/OWT scale, or ARPC without size-1 in the mixture.

## Pre-registered launch order (after 1500 readout)

**Directional only** — do not require low-t ≥ 0.90 or harness-PASS to launch A. Recipe-promotion AND stays separate (don’t promote A into the *shared default* until micro-go + health).

| If 1500 shows… | First move |
|----------------|------------|
| low-t moves **off 0** | Budget helps; optional longer neutral train; levers secondary |
| low-t still **hard 0** | Launch **A** (mixture) — BlockGen’s headline shared lever |
| Still flat after A | **B** (t-strat alone) *or* **C** (mix with 1) — pick one axis, not both |
| Still flat | Document budget/recipe-class limit — not more naked step rungs |

Always: `eval.t_bucketed_nll=true`, WandB `block_qwen_trials`. Never A+B in one job. Tax strata still need harness-PASS (then family-clears).
