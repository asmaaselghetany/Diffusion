# Baseline: AR → block × {masked, uniform} — BlockGen block skeleton

**Fixed:** AR Instruct → block (`LINE=ar2block`). That is the conversion family,
not a lever.

**Block skeleton (both arms):** BlockGen’s claim — one blockwise recipe, two
corruptions. Levers are ablations/combos on top of this, never silent defaults.

See [FAIR_AR2BLOCK_MAP.md](FAIR_AR2BLOCK_MAP.md) · [BLOCK_PATH.md](../BLOCK_PATH.md) ·
[LEVERS.md](LEVERS.md) · `configs/levers/registry.yaml`.

---

## Experimental contract (do not redefine)

| Layer | Baseline (= C0 / U0) | Levers (ablations) |
|-------|----------------------|--------------------|
| **Init (fixed)** | AR Instruct → block | — |
| **Block spine** | Same BlockTrainer / Qwen block / `block_size` / data / optim | mixture, ARPC, anneals, … |
| **Train hooks** | **All off**: no shift, complementary, plain_ce, fdllm schedule, mixture, ARPC, anneals | `C2_*`, `fastdllm_*`, `xfer_*`, … |
| **Decode** | Masked floor: packing + **conf remask** (`hierarchical` thr=0.9). Uniform open-loop: packing + **ancestral** (`hierarchical_ancestral`); remask twin = `uniform_commit` (UCC). `baseline`→`hierarchical` pins thr on both names but thr is **dead on Unif** without sticky | `hierarchical_ss`, `hierarchical_arpc`, `hubmatch`, `dual_cache`, `uniform_commit`, `hierarchical_quiet`, `hierarchical_ancestral` |
| **Arm difference only** | Absorbing vs uniform process (+ `subs`/`mean`, prior/posterior) | — |
| **Primary meters** | lm-eval under that shared ancestral decode | Same meters on lever cells for Δ |
| **Not baseline** | Fast-dLLM remask/DualCache claim, UCC, BlockGen TinyGSM T≈0.1/mixture scratch | Lever or other-family cells |

**What BlockGen shares that we share on the block path:**

- Same geometry hooks available (off by default on conversion floor)
- **Generate packing:** `[clean_prefix | noisy_block]` + `block_generation_mask`
  (`backbone.block_gen_logits`; BlockGen `forward_generate`) — wired in
  `BlockSampler` open-loop hierarchical path
- Masked conf remask (`hierarchical`); Unif open-loop ancestral
  (`hierarchical_ancestral`); Unif remask twin = `uniform_commit`
- ARPC as a **shared sampler add-on** (lever; profile `hierarchical_arpc` =
  B0 packing + ARPC at T=1; `hierarchical_quiet` = TinyGSM-ish T≈0.1 + ss)

**What changes with the arm (not a lever):** process, parameterization label,
prior / posterior / ARPC redraw kernel.

**Fairness rule:** `(lever cell) − (matched C0 or U0 floor)` under the same
decode unless the lever *is* a decode lever.

**Naming:** `C0` = masked floor · `U0` = uniform floor · `N0` = native floor.  
`fastdllm_*` / `B3_*` / `xfer_*` are **not** the conversion baseline.

**Claim ckpts for new decode bake-offs:** C0 `1762534` · U0 `1955203`+  
(Do **not** use U0 `1849335` as floor — see DESIGN_LOCKS `U0-1849335-CONTAMINATED`.)

---

## Phase-0 bake-off cells (decode-only)

Paper-faithful both-arm recipes + homemade twin probes.  
Submit: `./scripts/submit_baseline_bakeoff.sh`

| Cell | Profile | Steps | Role |
|------|---------|-------|------|
| **B1** | `hierarchical` | 32 | Contract floor (BlockGen ancestral) |
| **B2** | `hierarchical_arpc` | 32 | BlockGen ARPC on B0 packing |
| **B3** | `hierarchical_ss` | 32 | Hub packing + ancestral |
| **B4** | `hierarchical_arpc` | 8 | BlockGen `ar-then-arpc` few-step (not naked ancestral) |
| **DC** | `dual_cache` | 32 | Masked DualCache only (no U0 twin in bake) |
| **UC** | `uniform_commit` | 32 | Our UCC probe — **once**, uniform only |

**Scores:** [`BAKEOFF_2026-09-24.md`](BAKEOFF_2026-09-24.md) (valid/invalid marked).  
On disk: `Diffusion/outputs/block_qwen/ar2block_{masked_1762534,uniform_1955203}/lm_eval_bakeoff_20260924_*`

---

## Neutral comparison design

`block_qwen` trains two arms from the same AR-init checkpoint with **identical**
everything except corruption:

| Shared | Masked arm | Uniform arm |
|--------|------------|-------------|
| Qwen2.5 block backbone, `block_size`, seq len, data, optim, steps | `forward_process_name: masked` | `forward_process_name: uniform` |
| `configs/experiment/block_qwen.yaml` | `algo=block_masked` | `algo=block_uniform` |
| SUBS / Duo ELBO, log-linear noise, block sampler | absorbing per-block mask | uniform π = 1/V_eff per block |

Optional hooks default **off** in `configs/algo/block_*.yaml`. Enable only via
`./scripts/submit_lever.sh --preset …`.

## Shared conversion hygiene (not levers)

See `src/discrete_diffusion/data/conversion_baseline.py`:

- Train↔eval ChatML (conversion template)
- Hub-padded vocab keep (`151936`)
- Nemotron split defaults (math/code capped)

Decode **truncation** (`hierarchical_kv`) + BlockGen generate packing is skeleton
fairness. Confidence remask / DualCache / UCC are **decode levers**.

## Decode profiles (source of truth)

`src/discrete_diffusion/evaluations/decode_profiles.py` · allowlist in
`examples/block_qwen/lm_eval.sh`.

| Profile | Packing | Kernel | Notes |
|---------|---------|--------|-------|
| `baseline` → `hierarchical` | B0 `block_gen_logits` | conf remask thr=0.9 | Floor (B1) |
| `hierarchical_ss` | Hub single-stream | conf remask thr=0.9 | B3 |
| `hierarchical_ancestral` | B0 | mid-α ancestral thr=null | Forensic only |
| `hierarchical_arpc` | B0 | remask + ARPC (T=1) | B2 |
| `hierarchical_quiet` | Hub ss | ARPC + T≈0.1 | TinyGSM-ish lever |
| `hubmatch` | Hub ss + sub8 | conf remask | Masked |
| `dual_cache` | Hub ss + DualCache | conf remask thr=1 | Masked only |
| `uniform_commit` / `uniform_dual` | B1 pack thr=0.9 / DualCache pack thr=1 | Hub-clean UCC (no revise) | Uniform **UC** / DualCache twin; **ours** |
| `full_seq_dual` | full-seq | ancestral | Ablation only |

## Eval routing (baseline)

| Cell | `submit_family_eval` stack | Decode |
|------|----------------------------|--------|
| C0 / U0 (hooks off) | `conversion_lm_eval` / U0 `blockgen_arpc` + `--lm-eval-only` | `hierarchical` |
| Bake-off matrix | `submit_baseline_bakeoff.sh` | B1–B4 / DC / UC |
| Hub remask / DualCache lever | `FORCE_DECODE_PROFILE=hubmatch` / `fastdllm_lm_eval` | remask ± K/V |
| UCC lever | `FORCE_DECODE_PROFILE=uniform_commit` | USDM conf-commit |
| Native BlockGen | `blockgen_*` | ancestral ± ARPC |
| Fast-dLLM GitHub × our mix | `submit_hub_fastdllm_c2.sh` | their `eval.py` on that train |

Override with `FORCE_STACK` / `FORCE_DECODE_PROFILE` / `NUM_STEPS`.

## Masked / uniform code map

| Arm | FP + loss | Notes |
|-----|-----------|--------|
| Masked | `block_masked` + SUBS ELBO | Fast-dLLM train levers off by default |
| Uniform | `block_uniform` + Duo ELBO | BlockGen train levers off by default |

References: BlockGen [2606.02241](https://arxiv.org/abs/2606.02241) · Fast-dLLM [2509.26328](https://arxiv.org/abs/2509.26328) · UNI-D² infrastructure.
