# Levers registry (skeleton vs fluency)

Neutral `block_qwen` keeps these **off**. Add a row *before* enabling a lever
on a trial; promote to shared recipe only after a pre-registered go/no-go
(harness ∧ arm-sanity) on **both** corruptions when the lever claims to be
corruption-agnostic.

**Families (mandatory):**

| Family | `LINE` | Examples |
|--------|--------|----------|
| **conversion** | `ar2block` | `C0`, `C2_*`, `fdllm_*`, `C5_*`, `B4_*` |
| **native** | `block` | `N0`, `B3_*` (except t_strat), `blockgen_*` |
| **transfer** | `ar2block` + native knobs | `xfer_*` only — not a BlockGen claim |
| **shared** | either | decode `hierarchical_kv` / `dual_cache`; `t_strat_05` |

**Free sampling (adopted default):** `sample_mode=auto` + `decode_profile=baseline`
(see `.cursor/rules/sampling-family-policy.mdc`). Native → LM free-gen;
conversion → chat-header free-gen; bare BOS is ablation only. Exactness overlays
(DualCache+thr / ARPC) are opt-in, not free-gen defaults.

**Wiring:** use `./scripts/submit_lever.sh` →
[`configs/levers/registry.yaml`](../../configs/levers/registry.yaml) via
`tools/resolve_lever.py`. Do **not** hand-write lever Hydra flags. Wrong arm,
wrong family, conflicts, and missing prerequisites are refused.

```bash
./scripts/submit_lever.sh --list
# Conversion
./scripts/submit_lever.sh --preset fdllm --arm masked --dry-run
./scripts/submit_lever.sh --preset C2_fdllm_full --arm masked --paper
# Native / BlockGen
./scripts/submit_lever.sh --preset blockgen_uniform --arm uniform --paper
./scripts/submit_blockgen_owt.sh
# Transfer (explicit)
./scripts/submit_lever.sh --preset xfer_mixture --arm masked --paper
```

`submit_blockgen_lever_micro.sh` is **retired** (old path forced `ar2block`).

| Lever | Source | Family / arm | Config key | Individually tested? | Promoted to shared? | Notes |
|-------|--------|--------------|------------|----------------------|---------------------|-------|
| (none — skeleton) | — | both | defaults in `configs/algo/block_*.yaml` | Layer 0–3 unit tests | **yes** (neutral) | Correct-but-undertrained is OK |
| `shift_loss_targets` | Fast-dLLM v2 | **conversion + masked** | `algo.shift_loss_targets` | yes | **no** | Prefer shift-only micro before full fdllm |
| `complementary_masks` | Fast-dLLM v2 | **conversion + masked** | `algo.complementary_masks` (+ `complementary_batching=fused`) | yes | **no** | Paired m/~m; Hub fused 2B |
| `mask_schedule=fast_dllm` | Fast-dLLM v2 | **conversion + masked** | `algo.mask_schedule` | unit | **no** | `p_mask=(1-ε)t+ε` |
| `loss_weighting=plain_ce` | Fast-dLLM v2 Hub CE | **conversion + masked** | `algo.loss_weighting` | yes (denom tests) | **no** | Mask-site mean only; see **PLAIN-CE-DENOM** |
| `sub_block_size` | Fast-dLLM decode | shared / decode | `sampling.sub_block_size` | sampler | **no** | Pair with `hierarchical_kv` |
| `hierarchical_kv` | Fast-dLLM decode | shared / decode | `sampling.hierarchical_kv` | sampler | **no** | Truncate-only progressive |
| `dual_cache` | Fast-dLLM DualCache | shared / decode | `sampling.use_block_cache` | unit | **no** | Requires `hierarchical_kv` |
| `joint_ar` / `causal_clean` | NLD-style C5 | **conversion** | `algo.joint_ar_alpha` / `causal_clean_stream` | joint unit | **no** | `val/joint_nll` vs `val/nll` |
| `hybrid_p10` / `hybrid_p50` | B4 hybrid | **conversion + hybrid** | `algo.hybrid_p_uniform` | hybrid unit | **no** | `p10` pins default; `p50` sweep |
| `block_size_mixture` | BlockGen analogue | **native** (or `xfer_*`) | `algo.block_size_mixture` | unit | **no** | Uniform list draw |
| `block_weights` | BlockGen | **native** (or `xfer_*`) | `algo.block_weights` | geometry unit | **no** | Weighted `2^k`; XOR with list mixture |
| `block_size_per_gpu` | BlockGen | **native** (or `xfer_*`) | `algo.block_size_per_gpu` | geometry unit | **no** | `u-stratified` needs weights |
| `stratified_gamma` | our t-strat | shared | `algo.stratified_gamma` | with mixture | **no** | ≠ BlockGen `u-stratified` |
| `pure_noise_block_sizes` | BlockGen | **native** (or `xfer_*`) | `algo.pure_noise_block_sizes` | trainer | **no** | Force `t=1` at listed sizes |
| `loss_type_special_cases` | BlockGen | **native** (or `xfer_*`) | `algo.loss_type_special_cases` | trainer | **no** | e.g. `[1,ce]` |
| `use_arpc` | BlockGen §3.3 | **native + uniform** (or `xfer_*`) | `sampling.use_arpc` + `arpc_mode` | sampler | **no** | Needs size-1 in mix |

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
| **COMP-POLARITY** | Old `complementary_masks` flipped `~move_mask` in-place (wrong vs Fast-dLLM paired views). | Comp/both micros BPD ~24 | **fixed** — Hub fused 2B `m`/`~m` in `BlockTrainer.nll` (`complementary_batching=fused`) |
| **PLAIN-CE-DENOM** | `plain_ce` zeroed clean sites in the numerator but `_loss` divided by `valid_tokens.sum()` (all positions). Complementary doubles valid → ~½ Hub scale at same LR. | C2 ≪ C0 on same skeleton (~10 pp GSM8K) | **fixed** — mask-site denom (`_plain_ce_token_count`); tests in `test_block_shift_loss.py` |
| **SFT-ATTN-PROMPT** | `nll` passed assistant-only `valid_tokens` as backbone `attention_mask`, so SDPA blocked attending to user/system prompt. Hub train only uses structural block-diff mask (prompt stays visible; `labels=-100` drops prompt from CE). | Absolute Hub gap (GSM8K/IFEval) on **both** C0 and C2 | **fixed** — `attention_mask=batch['attention_mask']` for attn; assistant mask only for corrupt/loss; `tests/test_sft_attention_prompt.py` |
| **FAST-DLLM-DOUBLE-EPS** | `sample_block_timesteps` floored `t` to `[eps,1]` then `fast_dllm` applied `p=(1-ε)t+ε` again → min `p≈2ε` vs Hub `t~U(0,1)`. | Mild C2-only schedule bias | **fixed** — `fast_dllm` uses `sampling_eps=0` for `t`; test in `test_mask_schedule_elbo.py` |
| **HUB-STRUCT-ATTN** | Hub train overwrites attn with structural mask only; we still applied pad blocking → different softmax on packed MASK pads. | Grad diff on padded rows | **fixed** — `algo.hub_struct_attn_only` via `hub_train_parity` on C2_fdllm_full |
| **HUB-ANTITHETIC / ignore_bos** | Default antithetic + ignore_bos≠Hub | Mild C2 t-law / corrupt | **fixed** — antithetic off under `fast_dllm`; `ignore_bos=false` in `hub_train_parity` |

BlockGen-derived levers: [`BLOCKGEN_LEVERS.md`](BLOCKGEN_LEVERS.md) (**native
`LINE=block`**). Prefer `submit_lever.sh` / `submit_blockgen_owt.sh`. Legacy
`submit_blockgen_lever_micro.sh` / `submit_fdllm_lever_micro.sh` are retired or
avoided.
