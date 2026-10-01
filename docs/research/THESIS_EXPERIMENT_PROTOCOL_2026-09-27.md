# Thesis experiment protocol (preregistered) — 2026-09-27

**Status:** Preregistered **before** attack-train GSM harvest.  
**Harvest closed 2026-09-30:** `AR2B-U-floor = U0_ss_shift` `2092151` (ARPC 38.1%);
C3 `2104831` done but collapsed (ARPC 4.5%). Queue empty.  
Do not edit selection rules after opening those SUMMARYs without a version bump.

Companion: [`AR2BLOCK_GSM_TABLE.md`](AR2BLOCK_GSM_TABLE.md) · [`DESIGN_LOCKS.md`](DESIGN_LOCKS.md) · [`PAPER_EXPERIMENTS.md`](PAPER_EXPERIMENTS.md)

---

## Thesis question (abstract)

> How does the choice of **training paradigm** and **representation** affect
> converting autoregressive Instruct models into alternative generation
> procedures, and which **individual** training/design factors account for
> observed differences?

Experimental decomposition:

| Question | Family | Purpose |
|----------|--------|---------|
| Can AR→**block** conversion be made fair on both corruptions? | AR→block | Main investigation |
| How does **full-seq diffusion** compare (Unifusion-style)? | AR→full-seq = **C3** | Paradigm contrast |
| How does **staying causal AR** compare (same budget)? | **`A_ar_sft`** | Control / reference |
| What if AR + block objectives are **joint**? | AR+block = C5 | Alternative paradigm |
| How different is **native** block training? | Native / BlockGen | Paradigm contrast |
| Which **single** choices matter? | Layer C axes | Sensitivity |

**Rule:** “full-seq” = **diffusion only** (C3). Causal AR is never called full-seq.

---

## Three-layer methodology (locked)

```
Layer A  Lock a named fair FLOOR per paradigm
Layer B  Search mixes only inside that paradigm (discovery; optional)
Layer C  Freeze the floor → change ONE lever family → measure Δ
```

| Concept | Role | May become floor? |
|---------|------|-------------------|
| **Provisional / named floor** | Scientific control for axis tables | Yes (by selection rule) |
| **Attack / search cell** | Discover robust recipes | Only if selection rule says so |
| **Axis cell** | `floor + {one lever family}` | No — report Δ only |
| **Final configuration** | Optional post-axis recipe for systems demo | Separate from floor claims |

**Selection conditioning (acknowledged):**  
Axes are conditioned on the selected floor. Thesis claim is therefore:

> *We use a held-fixed selection protocol to establish a fair operating point
> within each paradigm; sensitivity experiments then quantify individual
> design choices relative to that point — not “we found the globally best model.”*

---

## Floor-selection rules (preregistered)

### AR→block (gate — main bottleneck)

**Provisional floor (search reference, already scored):**  
hard `xfer_bg_mix_32_blockgen` — masked `2020048` ∥ uniform `2020049`  
ARPC s32 GSM: **55.6 / 33.4** (gap **22.2** pp). Ancestral ~2 / ~5.

**Candidates (discovery):** provisional +  
`U0_ss_shift` `2092151` · `blockgen_ss` `2092154` · `blockgen_ss_shift` `2092155` ·  
`blockgen_unifv` `2094996`/`2092156`.  
Meters: `hierarchical_ancestral` + `hierarchical_arpc` s32 only. **UCC/UDual out.**

**Selection algorithm (apply once, write `AR2B-U-floor = …`):**

1. **Primary:** maximize **uniform** `hierarchical_arpc` GSM (flex-extract).  
2. **Constraint A:** Unif ARPC ≥ provisional Unif ARPC − **1.0** pp.  
3. **Constraint B (paired masked):** if the candidate has a matched masked twin
   under the same train recipe, masked ARPC ≥ **55.6 − 3.0** pp; else compare
   Unif-only candidates against provisional Unif and keep hard-M `2020048` as
   the masked floor twin for gap reporting.  
4. **Tie-break (secondary):** smaller |M−U| ARPC gap; then higher Unif ancestral.  
5. **Fallback:** if no attack satisfies A (+ B when applicable) → **retain provisional
   hard floor**. Attacks stay in a **discovery** table only.  
6. **Never** select on UCC, DualCache, quiet, or GenPPL alone.

After selection: **no new xfer mashups** until Layer C axes on that floor are done.

### AR→full-seq diffusion (C3)

**Floor:** first successful paper `C3_fullseq` run (`2096579` in flight) —
uniform + shift, `block_size=2048`.  
**Meters:** full generative suite (`mmlu_generative,gsm8k,ifeval`); GenPPL+H hygiene.  
**Axes later:** one lever family at a time (ss packing if OOM, schedule, …).

### Matched causal AR (`A_ar_sft`)

**Floor (locked):** `ar_sft_1857323` — GSM **70.8** / IFE **24.6** / MMLU **48.4**.  
No retrain. Axes only if Tab-1 needs sensitivity (usually none).

### AR+block (C5)

**Floor:** one named joint cell (`C5_joint_ar` *or* `C5_causal_clean` — pick
**before** launching axes; default prefer `C5_joint_ar` if already scored).  
**Meters:** `val/nll` (diffusion) + `val/joint_nll` + generative suite.

### Native / BlockGen

**Contrast only** — one B3/OWT-style cell. Not the AR→block floor. No mashup into C0/U0.

---

## Decision meters by paradigm

| Paradigm | Primary | Secondary | Forbidden as floor meter |
|----------|---------|-----------|---------------------------|
| AR→block | ancestral + ARPC s32 full suite (GSM+IFE+MMLU) | GenPPL+H | UCC / UDual / quiet / GSM-only |
| C3 full-seq diffusion | full generative suite | GenPPL+H (hygiene) | Block remask “B1” contract |
| `A_ar_sft` | GSM / IFE / MMLU | — | Diffusion profiles |
| C5 joint | generative + joint_nll | diffusion nll | Comparing joint_nll to C0 nll as identical |
| Native | generative / ELBO per family contract | — | Tagging as conversion |

---

## One-page experiment matrix

| ID | Family | Floor / role | Delta | Primary metric | Secondary | Selection / note | Status |
|----|--------|--------------|-------|----------------|-----------|------------------|--------|
| F-AR2B-hard | AR→block | **provisional floor** | hard mix + x0_causal | ARPC GSM M∥U | ancestral; gap | Retain unless rule promotes attack | Scored 55.6/33.4 |
| F-AR2B-soft | AR→block | discovery | soft size-1 | ARPC | ancestral | Not floor (Unif worse) | Scored 46.8/18.7 |
| A-ss | AR→block | attack | +ss packing on hard | ARPC / anc | gap | Rule §§1–5 | Train `2092154` |
| A-ss_shift | AR→block | attack | +ss + shift on hard | ARPC / anc | gap | Rule §§1–5 | Train `2092155` |
| A-U0_ss_shift | AR→block | **`AR2B-U-floor`** | U0 + ss + shift | ARPC / anc | — | Rule §§1–5 **PASS** Unif ARPC 38.1 | Train `2092151` |
| A-unifv | AR→block | attack | +Unif(V) on hard | ARPC / anc | gap | Rule §§1–5 | Resume `2094996` |
| C-udual | AR→block | control only | uniform_dual decode | GSM | IFE | **Never floor** | Done 46.9% |
| F-C3 | full-seq diff | **C3 floor** | bs=2048 + shift | GSM/IFE | GenPPL+H | First good `C3_fullseq` | Train `2096579` |
| F-AR | causal AR | **`A_ar_sft` floor** | — | GSM/IFE/MMLU | — | Locked | `1857323` |
| F-C5 | joint | C5 floor | joint_ar (default) | gen + joint_nll | nll | Pick before axes | TBD / existing |
| F-N0 | native | contrast | scratch block | per native contract | — | Separate table | optional |
| X-* | any | **axis** | floor + one lever | paradigm primary | — | Only after floor named | blocked until gate |

Fill `AR2B-U-floor = …` in [`AR2BLOCK_GSM_TABLE.md`](AR2BLOCK_GSM_TABLE.md) when the rule fires.
**Filled 2026-09-28:** `AR2B-U-floor = U0_ss_shift` `2092151` (Unif ARPC 38.1).

---

## Scope lock

This matrix is **enough** for a master’s thesis.  
Resist new paradigms / xfer mashups until the AR→block gate closes and Layer C
has at least one clean axis table on the named floor.
