# Fair AR→block design map (both arms)

**Family lock:** `LINE=ar2block` (Qwen Instruct → block SFT).  
**Not in this map:** `LINE=block` scratch DiT, TinyGSM native B3 claim cells, xfer package as silent U0 default.

**Living docs:** `BASELINE.md` · `DESIGN_LOCKS.md` · `LEVERS.md`  
**Submit bake-off:** `./scripts/submit_baseline_bakeoff.sh`

Goal: paper-faithful both-arm floors from Fast-dLLM / MDLM / Duo·UDLM / BlockGen, bake off, then ablate. Homemade twins (UCC / DualCache) are scored **beside** the paper floors, not as the contract default.

---

## What is fixed vs what may vary

| Axis | Fixed for fair floor | May differ |
|------|----------------------|------------|
| Init | AR Instruct → block | — |
| Backbone / data / length / steps / GBS | Shared | — |
| Corruption | — | **absorb vs uniform** (the arm) |
| Train ELBO family | Arm-matched (SUBS vs Duo) | Hooks (shift, complementary, CE@1, …) = levers |
| Generate packing | Must not attend future noise | BlockGen `[prefix\|block]` vs Hub ss |
| Reverse kernel | Matched across arms when claiming “both” | Ancestral / ARPC / remask / NFE |

**Floor bake-off is decode-only** on fixed C0/U0 ckpts — no scratch retrain.

---

## Layer A — Train objective (hooks off for floor)

| Cell | Masked | Uniform | Paper DNA | Fair both-arm? |
|------|--------|---------|-----------|----------------|
| **A0** ELBO | SUBS / absorbing ELBO | Duo / UDLM-style uniform ELBO | MDLM + Duo + BlockGen `loss_type=elbo` | **Yes — floor train** |
| A1 Hub plain CE | `plain_ce` on masks | no twin | Fast-dLLM Hub train CE | Masked-only lever |
| A2 size-1 CE | `loss_type_special_cases` | same | BlockGen CE@1 | Lever / xfer |

---

## Layer B — Generate packing

| Cell | Packing | Both arms? | Paper DNA | Wire today | Role |
|------|---------|------------|-----------|------------|------|
| **B0** BlockGen generate | `[clean_prefix \| noisy_block]` + `block_generation_mask` | Yes | BlockGen `forward_generate` | `backbone.block_gen_logits` under `hierarchical` | **Floor packing (contract)** |
| B1 Truncated dual | `concat(xt[:A], x0[:A])` shared RoPE | Yes | Our dual-train approx of B0 | hierarchical fallback if no `block_gen_logits` | Fair packing A/B |
| B2 Hub single-stream | `xt[:A]` + `eval_block_diff_mask` | Yes | Fast-dLLM / Hub generate | `hierarchical_ss` | Fair packing A/B |
| B3 Full-seq dual | Attend future MASK/Unif | **Illegal** open-loop | GSM 2.27% footgun | `full_seq_dual` only | Out of scope |

---

## Layer C — Reverse kernel / decode

| Cell | Masked | Uniform | Paper DNA | Profile | Role |
|------|--------|---------|-----------|---------|------|
| **C0** Mid-α ancestral | Absorbing mid-α + last `α_s=1` | Uniform mid-α + last `α_s=1` | BlockGen / MDLM / Duo ancestral | `baseline` → `hierarchical` | **Floor kernel (contract)** |
| C1 Conf remask twin | Hub thr≈0.9 greedy sub8 | UCC (`uniform_commit`) | Fast-dLLM remask; **UCC is ours** | `hubmatch` ∥ `uniform_commit` | Homemade twin (not paper floor) |
| C2 DualCache remask | C1 + `use_block_cache` thr=1 | **no twin** (MASK-only) | Fast-dLLM DualCache | `dual_cache` | Masked lever only |
| C3 Quiet ancestral | C0 + T≈0.1 ± ARPC + ss | same | BlockGen TinyGSM temps | `hierarchical_quiet` | Lever |
| C4 ARPC on B0 packing | re-MASK low-LL | re-Unif low-LL | BlockGen §3.3 | **`hierarchical_arpc`** | Paper-faithful ARPC (T=1, not quiet) |

---

## Phase 0 — Paper-faithful baselines (B1–B4) + homemade twins

Train = **A0**. Same ckpts: **C0 `1762534`**, **U0 `1955203`** (not contaminated `1849335`).  
Submit: `./scripts/submit_baseline_bakeoff.sh` → outs `lm_eval_bakeoff_20260924_*`.

**Scored table (valid / invalid):** [`BAKEOFF_2026-09-24.md`](BAKEOFF_2026-09-24.md) — canonical bake scoreboard.

### Paper-identical / parallel (both arms)

| # | Name | Profile (both arms) | Paper DNA | Identical? |
|---|------|---------------------|-----------|------------|
| **B1** | Block ancestral / conf floor | `hierarchical` s=32 | BlockGen packing; masked thr=0.9 | C0 B1 = **56.2%** conf; U0 remask twin = **UC** (not mid-α) |
| **B2** | Block ARPC | `hierarchical_arpc` s=32 | BlockGen §3.3 on B0 packing | **Identical** sampler; redraw follows π — **both arms valid** |
| **B3** | Hub packing | `hierarchical_ss` s=32 | Fast-dLLM generate geometry | C0 B3 = **56.3%**; do not cite `*.COERCED_UCC` |
| **B4** | Few-step ARPC | `hierarchical_arpc` s=8 | BlockGen `ar-then-arpc` (`steps: 8`) | **Both arms valid** |

### Homemade / asymmetric (validity probes)

| Tag | Masked | Uniform | Notes |
|-----|--------|---------|-------|
| **DC** | `dual_cache` thr=1 greedy | **omit** (no K/V twin) | C0 only in bake submit |
| **UC** | — | `uniform_commit` once | **Our** UCC — sole UCC baseline |

**Interpretation**

1. Lock **B1** as scientific floor unless bake-off overturns it (C0 B1 = 56.2% conf).  
2. B2/B3/B4 = matched ablations (ARPC / packing / NFE).  
3. DC/UC cite Δ to B1; never redefine floor as UCC∥DualCache.  
4. If B1/B2 split by arm → report two floors; no cross-kernel lever Δ.  
5. Cite U0 numbers only from the **Valid** table in `BAKEOFF_2026-09-24.md`.

---

## Phase 1 — Ablations after floor lock

| Bucket | Cells | Arms | Paper |
|--------|-------|------|-------|
| Fast-dLLM train | shift, complementary, fast_dllm schedule, plain_ce | masked | Fast-dLLM v2 |
| Fast-dLLM decode | sub_block, DualCache, thr/greedy, hubmatch | masked; UCC = uniform conf twin | Fast-dLLM |
| BlockGen geometry | mixture / weights / u-strat / pure_noise / CE@1 | both via `xfer_*` | BlockGen |
| BlockGen sampler | quiet T, ARPC on size-1 mix | both | BlockGen |
| Full-seq → block ports | Duo curriculum/distill; Schiff CFG on U0 | U0 first | Duo / Simple Guidance |
| Paradigm extras | hybrid_p*, joint_ar / causal_clean | hybrid / joint | B4 / C5 |

---

## Paper → what it owns

| Paper | Masked | Uniform | Fair both-arm use |
|-------|--------|---------|-------------------|
| D3PM / SEDD | Absorbing rate | Uniform rate | Same CTMC framework → A0 + arm π |
| MDLM / SUBS | Absorbing ELBO | — | Train A0 masked |
| Duo / UDLM | — | Uniform ELBO + guidance; few-step needs distill/ARPC not naked NFE cut | Train A0 uniform; **B4** = BlockGen ARPC s=8 |
| Fast-dLLM | AR→block, remask, DualCache | no DualCache twin | Init; packing **B3**; remask/DC = levers |
| BlockGen | Absorb + ancestral + ARPC | Uniform + ancestral + ARPC | **B1** skeleton; **B2** ARPC |
| LLaDA / Hub scripts | conf remask pins | — | hubmatch pins |

---

## External correctness: Fast-dLLM GitHub × our mix

**Intent (one chain):** vendored `third_party/Fast-dLLM/v2` train on our C2 Nemotron mix → **same** run → their `eval.py`.  
**Script:** `./scripts/submit_hub_fastdllm_c2.sh`  
**Root:** `/e/scratch/scifi/elsayed3/hub_fastdllm_c2/`  
**Status:** prep OK; early trains failed (`wandb`); resubmitted with `--report_to none`, post-train `modeling.py` copy, IPv4 master (jobs `1996142`→`1996144`).  

Separate check already done: **our** weights in Hub eval ≈ our sampler → residual Hub gap looks like **data/mix**, not decode porting.

---

## Illegal / asymmetric (do not call “fair both-arm”)

| Combo | Why |
|-------|-----|
| hubmatch ∥ DualCache as U0 floor | DualCache is MASK-only |
| hubmatch masked + ancestral uniform as matched pair | Protocol asymmetry |
| Full-seq dual either arm | Future-noise attend |
| Uniform greedy / `argmax(q_xs)` | Locks Unif prior |
| xfer TinyGSM as U0 floor | Geometry confound |
| UCC as “paper method” | **Ours** — cite as construction |
| Cite U0 `1849335` as floor | Contaminated train contract |

---

## Profile ↔ cell cheat sheet

| Profile | Cell |
|---------|------|
| `hierarchical` / `baseline` | B1 (s=32) |
| `hierarchical_arpc` | B2 (s=32) · B4 (s=8, BlockGen few-step) |
| `hierarchical_ss` | B3 |
| `hierarchical_quiet` | TinyGSM-ish quiet lever (ss + T=0.1 + ARPC) |
| `hubmatch` | Fast-dLLM remask (masked) |
| `dual_cache` | DualCache (masked only) |
| `uniform_commit` / `uniform_dual` | UCC (**UC cell only**) |

See also: `BASELINE.md`, `LEVERS.md`, `DESIGN_LOCKS.md`, `UNIFORM_CONFIDENCE_COMMIT_2026-09-24.md`.
