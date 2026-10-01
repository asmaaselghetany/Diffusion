# Hard audit (forensic): why U0 GSM ≈ 5–6% — deep pipeline failures

**Date:** 2026-09-21 · **Queue:** empty · **Do not re-queue until locked**

Companions: `HARD_AUDIT_UNIFORM_PIPELINE_2026-09-21.md`,
`DEBUG_U0_UNIFORM_SKELETON_2026-09-21.md`.

---

## Verdict

The published U0 GSM **6.06% flexible / 0% strict** is **not** a fair measure of
uniform AR→block capability. It was produced under a **broken decode protocol**.
C0’s ~63% used DualCache/hubmatch; U0’s ~6% used **baseline full-seq Unif**.

**~5–6% ≈ digit-scrape floor** on fluent-but-wrong CoT (strict `####` never
fires). Same flexible scorer gives C0 ~63% — the gap is generation quality under
hostile geometry, not the harness.

---

## Smoking-gun evidence (this pass)

### 1. GSM run config (`lm_eval_gsm8k_fast/SUMMARY.json`)

| Pin | U0 value | Required for uniform |
|-----|----------|----------------------|
| `decode_profile` | **`baseline`** | `hierarchical` (BlockGen: prefix+active) |
| `hierarchical_kv` | **`false`** | `true` |
| `force_greedy` | **`1`** | `0` (argmax locks prior Unif) |
| `use_arpc` | false | optional (BlockGen TinyGSM often uses PC) |

Log line at init:

```text
[decode_profile=baseline] unmask_threshold=None hierarchical_kv=False
  use_block_cache=False single_stream=False use_arpc=False
```

Plus repeated warnings: `sampling.greedy=true is invalid for uniform reverse…
Forcing greedy=false`.

### 2. What the model actually wrote (GSM)

Logged assistant sample is **English-fluent, algebraically wrong**, no `####`:

- Invents steps, wrong products (`5 × 100 = $520`), truncates mid-sentence.
- Flexible-extract can still luck into a number → **~80/1319 ≈ 6%**.
- Strict-match **0.0** (C0 also has strict 0.0 — format is not the differentiator).

### 3. GenPPL samples under the same baseline protocol

`eval/unifusion_hygiene/nfe_sweep/steps_32/samples.txt`:

- Early: coherent English.
- ~char 3500+: **multilingual token soup** (`く möglich… HDDToken…`).
- `first_chunk_only` GenPPL ~61 **citeable**; full continuation ~76–92.
- **first_chunk hid mid-seq collapse.**

### 4. C0 contrast (fairness)

| Cell | Profile | GSM flexible |
|------|---------|--------------|
| C0 `1762534` | dual_cache / hubmatch thr=1 | **62–65%** |
| U0 `1849335` | **baseline** ancestral | **6.06%** |

Apples ≠ oranges. A C0 **baseline ancestral** control was never run.

---

## Ranked deep problems (pipeline)

| # | Problem | Status | Impact on ~5% GSM |
|---|---------|--------|-------------------|
| **P0** | Full-seq dual-stream attends **future Unif(V)** junk | Fixed in code (auto-hier + lm-eval re-force + **generate() re-assert** + profile coerce). **Old metric still from broken run.** | Dominant |
| **P0** | `DECODE_PROFILE=baseline` / ifeval / GenPPL hygiene **defaulted baseline** for U0 | Scripts remapped → `hierarchical` | Recurrence risk |
| **P1** | first_chunk GenPPL marked citeable while full text collapses | Hygiene dual mode exists; thesis must cite **full** + samples | False green light |
| **P1** | Old ckpt `1849335` = `Unif(V)` train; WT = `Unif(V\{MASK})` + ELBO `V_eff` | Decode re-eval ≠ new train contract | Retrain for lock |
| **P2** | Dual-stream `concat(xt,x0)` ≠ BlockGen `forward(x0=prefix, xt=block)` | Accepted residual; hierarchical is Qwen mitigation | Medium |
| **P2** | No ARPC / low-T on uniform Instruct (BlockGen TinyGSM often T≈0.1 + PC) | Not yet | Medium after geometry |
| **P3** | ChatML/EOS still in Unif support (rare) | Optional ban | Low |
| **P3** | Hub padded vocab ids in Unif (~0.18%) | Document | Low |

---

## What is *not* the root cause

- Scorer / flexible-extract (C0 uses the same path at ~63%).
- “Uniform is inherently ~5%” (unsupported until hierarchical re-eval).
- ELBO sign (matches BlockGen; minimize-oriented `coeff<0`).
- Train val/ppl alone (can look fine while decode geometry is wrong).

---

## Code locks added this pass

1. `coerce_profile_for_forward`: `baseline` → `hierarchical` for uniform/hybrid.
2. `BlockSampler.generate()`: re-enable `hierarchical_kv` if demoted after `__init__`.
3. lm-eval: coerce profile before pins; strip greedy on uniform profiles.
4. `submit_ifeval_samples.sh` U0 → hierarchical; GenPPL hygiene default → hierarchical.

---

## Before any queue (unchanged gate)

1. Choose **(A)** hierarchical decode-only re-eval of old U0 (GSM smoke), or **(B)** retrain U0 under locked `V_eff` then eval.
2. Cite (A) as “old weights + fixed decode”, never as new train cell.
3. Do not cite first_chunk GenPPL without full-continuation + sample audit.
4. Optional: C0 baseline ancestral control for fairness table.

**Recommendation:** (A) GSM-only hierarchical smoke first; if needle moves, (B) for paper lock.
