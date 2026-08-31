# Levers registry (skeleton vs fluency)

Neutral `block_qwen` keeps these **off**. Add a row *before* enabling a lever
on a trial; promote to shared recipe only after a pre-registered go/no-go
(harness ∧ arm-sanity) on **both** corruptions when the lever claims to be
corruption-agnostic.

**Wiring (mandatory):** use `./scripts/submit_lever.sh` which resolves
overrides from [`configs/levers/registry.yaml`](../../configs/levers/registry.yaml)
via `tools/resolve_lever.py`. Do **not** hand-write `algo.complementary_masks`
etc. into `HYDRA_OVERRIDES` for design-cell micros — the registry refuses wrong arm,
conflicts, and missing prerequisites.

```bash
./scripts/submit_lever.sh --list
./scripts/submit_lever.sh --preset fdllm --arm masked --dry-run
./scripts/submit_lever.sh --levers shift --arm masked
./scripts/submit_lever.sh --preset blockgen_uniform --arm uniform
```

| Lever | Source | Agnostic? | Config key | Individually tested? | Promoted to shared? | Notes |
|-------|--------|-----------|------------|----------------------|---------------------|-------|
| (none — skeleton) | — | — | defaults in `configs/algo/block_*.yaml` | Layer 0–3 unit tests; Layer 4 pending clean `t_bucketed` micro | **yes** (neutral) | Correct-but-undertrained is OK |
| `shift_loss_targets` | Fast-dLLM v2 | **masked-only** | `algo.shift_loss_targets` | yes (`test_block_shift_loss.py`) | **no** | Needs `sampling` align (auto when shift on). Prefer shift-only micro before full fdllm. |
| `complementary_masks` | Fast-dLLM v2 | **masked-only** | `algo.complementary_masks` | yes (`test_complementary_masks.py`) | **no** | **Paired m/~m**; sequential dual forward (not polarity flip; not 2B batch — OOM-safe at seq 2048) |
| `mask_schedule=fast_dllm` | Fast-dLLM v2 | **masked-only** | `algo.mask_schedule` | unit (FP + ELBO α_eff) | **no** | `p_mask=(1-ε)t+ε`; ELBO/decode use `α_eff=1-p` |
| `sub_block_size` | Fast-dLLM decode | decode | `sampling.sub_block_size` | sampler path | **no** | Windowing; pair with `hierarchical_kv` |
| `hierarchical_kv` | Fast-dLLM decode | decode | `sampling.hierarchical_kv` (+ `sub_block_size`) | sampler unit | **no** | Truncate-only progressive decode |
| `dual_cache` | Fast-dLLM DualCache | decode | `sampling.use_block_cache` | DualCache unit | **no** | Requires `hierarchical_kv`; K/V approx |
| `joint_ar` / `causal_clean` | NLD-style C5 | shared | `algo.joint_ar_alpha` / `causal_clean_stream` | joint unit | **no** | `val/joint_nll` vs diffusion `val/nll` |
| `hybrid_p10` / `hybrid_p50` | B4 hybrid | **hybrid** | `algo.hybrid_p_uniform` | hybrid unit | **no** | `p10` pins default; `p50` is a real sweep |
| `block_size_mixture` | BlockGen analogue | yes (both) | `algo.block_size_mixture` | unit (divisibility) | **no** | Uniform list draw |
| `block_weights` | BlockGen | yes (both) | `algo.block_weights` | geometry unit | **no** | Weighted `2^k`; XOR with list mixture |
| `block_size_per_gpu` | BlockGen | yes | `algo.block_size_per_gpu` | geometry unit | **no** | `same\|random\|u-stratified`; requires `block_weights` |
| `stratified_gamma` | our t-strat (≠ BlockGen `u-stratified`) | yes | `algo.stratified_gamma` | with mixture | **no** | See `BLOCKGEN_LEVERS.md` |
| `pure_noise_block_sizes` | BlockGen | yes | `algo.pure_noise_block_sizes` | trainer validate | **no** | Force `t=1` at listed sizes |
| `loss_type_special_cases` | BlockGen | yes | `algo.loss_type_special_cases` | trainer validate | **no** | e.g. `[1,ce]` |
| `use_arpc` | BlockGen §3.3 | decode; **uniform-only** | `sampling.use_arpc` + `arpc_mode` | sampler unit | **no** | `simplified` or `blockgen` (+ corruption modes); requires size-1 |

## How to add a lever

1. Row in this table (status: not tested).
2. Implement behind a default-**false**/null config flag (do not change neutral defaults).
3. Add entry to `configs/levers/registry.yaml` (arms, conflicts, requires).
4. Unit test + optional micro with `eval.t_bucketed_nll=true` via `submit_lever.sh`.
5. Pre-register go/no-go; mark tested y/n.
6. Promote only if both arms (when agnostic) clear conjunction — else keep Track-2 / arm-specific.

See also: `BLOCK_QWEN_TRAINING.md`, `DESIGN_LOCKS.md`, `LOSS_SPECIAL_CASE_POLICY` in `block_trainer.py`.

## Known pipeline TODOs (not lever work)

| ID | Issue | Impact | Status |
|----|-------|--------|--------|
| **HYDRA-CKPT-MUL** | Loading a saved Lightning/`OmegaConf` ckpt config into `get_dataloaders` fails with `Unsupported interpolation type mul` on `trainer.accumulate_grad_batches` (and similar `${mul:…}` / `${div_up:…}` keys). Hit by `run_copy_x0_probe.py --real-batch`. | Real-batch diagnostics from ckpt config break; synthetic path OK. | **fixed** — `register_config_resolvers()` + `OmegaConf.resolve` in `run_copy_x0_probe.py` / `run_block_arm_sanity.py` before dataloader/model use |
| **TBUCKET-SHIFT-TRIM** | `_log_t_bucketed_nll` × `shift_loss_targets` → loss T-1 vs valid T (511 vs 512). Crashed Track 2 `138103` after step 500. | End-of-val logging only; ckpt at 500 may still be usable. | **fixed** in `block_trainer.py` + `test_t_bucketed_nll_shift_trim_aligns` |
| **COMP-POLARITY** | Old `complementary_masks` flipped `~move_mask` in-place (wrong vs Fast-dLLM paired views). | Comp/both micros BPD ~24 | **fixed** — sequential paired `m`/`~m` forwards in `BlockTrainer.nll` |

BlockGen-derived shared levers: [`BLOCKGEN_LEVERS.md`](BLOCKGEN_LEVERS.md). Prefer `submit_lever.sh` over `submit_blockgen_lever_micro.sh` / `submit_fdllm_lever_micro.sh` (legacy).
