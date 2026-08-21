# Levers registry (skeleton vs fluency)

Neutral `block_qwen` keeps these **off**. Add a row *before* enabling a lever
on a trial; promote to shared recipe only after a pre-registered go/no-go
(harness ∧ arm-sanity) on **both** corruptions when the lever claims to be
corruption-agnostic.

| Lever | Source | Agnostic? | Config key | Individually tested? | Promoted to shared? | Notes |
|-------|--------|-----------|------------|----------------------|---------------------|-------|
| (none — skeleton) | — | — | defaults in `configs/algo/block_*.yaml` | Layer 0–3 unit tests; Layer 4 pending clean `t_bucketed` micro | **yes** (neutral) | Correct-but-undertrained is OK |
| `shift_loss_targets` | Fast-dLLM v2 | **masked-only** | `algo.shift_loss_targets` | yes (`test_block_shift_loss.py`) | **no** | Track 2 `138103`: pos VR 1.0→0.75 at 500 steps; low-t still 0. **NO-GO**; do not promote |
| `complementary_masks` | Fast-dLLM v2 | **masked-only** | `algo.complementary_masks` | partial (FP path) | **no** | Paired with shift on `138103`; same NO-GO |

| `block_size_mixture` | BlockGen | yes (both) | `algo.block_size_mixture` | unit (divisibility) | **no** | Scratch-uniform recipe candidate; needs Layer-4 micro |
| `stratified_gamma` | our t-strat (≠ BlockGen `u-stratified` block-size) | yes | `algo.stratified_gamma` | with mixture | **no** | See `BLOCKGEN_LEVERS.md`; secondary to mixture |
| `use_arpc` | BlockGen §3.3 | decode; uniform-oriented | `sampling.use_arpc` | sampler unit | **no** | Decode-only; best with mixture incl. block size 1 |

## How to add a lever

1. Row in this table (status: not tested).
2. Implement behind a default-**false** config flag (do not change neutral defaults).
3. Unit test + optional micro with `eval.t_bucketed_nll=true`.
4. Pre-register go/no-go; mark tested y/n.
5. Promote only if both arms (when agnostic) clear conjunction — else keep Track-2 / arm-specific.

See also: `BLOCK_QWEN_TRAINING.md`, `LOSS_SPECIAL_CASE_POLICY` in `block_trainer.py`.

## Known pipeline TODOs (not lever work)

| ID | Issue | Impact | Status |
|----|-------|--------|--------|
| **HYDRA-CKPT-MUL** | Loading a saved Lightning/`OmegaConf` ckpt config into `get_dataloaders` fails with `Unsupported interpolation type mul` on `trainer.accumulate_grad_batches` (and similar `${mul:…}` / `${div_up:…}` keys). Hit by `run_copy_x0_probe.py --real-batch`. | Real-batch diagnostics from ckpt config break; synthetic path OK. | **fixed** — `register_config_resolvers()` + `OmegaConf.resolve` in `run_copy_x0_probe.py` / `run_block_arm_sanity.py` before dataloader/model use |
| **TBUCKET-SHIFT-TRIM** | `_log_t_bucketed_nll` × `shift_loss_targets` → loss T-1 vs valid T (511 vs 512). Crashed Track 2 `138103` after step 500. | End-of-val logging only; ckpt at 500 may still be usable. | **fixed** in `block_trainer.py` + `test_t_bucketed_nll_shift_trim_aligns` |

BlockGen-derived shared levers (draft, **do not launch** until 1500 readout): [`BLOCKGEN_LEVERS.md`](BLOCKGEN_LEVERS.md), `scripts/submit_blockgen_lever_micro.sh`.
