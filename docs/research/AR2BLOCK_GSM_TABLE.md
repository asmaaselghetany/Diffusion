# AR→block GSM — same Nemotron (masked ∥ uniform)

**Code version (forward_process utils):** SHA256
`1a3f6c494ed8bee8ea63fb66d7ff2b3e3f00cb3bf13555a3cb877749d1202f9e`
(276-line full file; tag `baseline-ar2block-utils-full-2026-10-01`). Eval JSON
must include `code_fingerprint.forward_process_utils.sha256`.

**Success bar (user lock):** ancestral + ARPC must work on **both** arms the way
they do for BlockGen / the way remask+ARPC already work on **masked**.  
UCC is a homemade probe only — **not** proof the pipeline is healthy.

**Floor protocol:** provisional hard M/U `2020048`∥`2020049`; attack promotion
only via preregistered rule in
[`THESIS_EXPERIMENT_PROTOCOL_2026-09-27.md`](THESIS_EXPERIMENT_PROTOCOL_2026-09-27.md)
(written **before** attack GSM).

**`AR2B-U-floor = U0_ss_shift` `2092151`** — Unif ARPC **38.1** (≥ hard-U 33.4 − 1).  
Masked twin for gap: hard-M `2020048` ARPC **55.6** (no matched masked twin under
the same U0+ss+shift recipe). Δ(M−U) ARPC **≈17.5** pp (was 22.2 on hard floor).
Ancestral @ **T=1** is unusable (~2–5% GSM; hard twins **2.4% M / 5.4% U**);
**T=0.1 open-loop 2×2 pending** — do not treat collapse as “just knobs” or as
a permanent open-loop verdict until that readout. Remask ~56% on same M
weights shows **masked** needs a corrector; Unif story open. Matched gap
floor = **ARPC** (UCC = homemade).

## Matched baseline (ARPC∥ARPC) — preregistered 2026-10-01

**Parallel enough to be the baseline**, with caveats below. Same profile ≠
same experiment; report both rows and the audit fields.

### Two-row reporting (locked)

| Row | Pair | GSM | Δ(M−U) | Reads as |
|-----|------|----:|-------:|----------|
| **Matched twin** | `2020048` ∥ `2020049`, `hierarchical_arpc` T=1 | **55.6** vs **33.4** | **22.2** | Same recipe, untuned Unif |
| **Tuned-Unif reference** | `2020048` vs `2092151`, same decode | **55.6** vs **38.1** | **≈17.5** | Best Unif recipe under same decode — **not matched recipe** |

The gap of interest sits between those two numbers. Never cite 55.6 vs 38.1
as a same-recipe twin. No masked `U0_ss_shift` twin exists yet.
`2092151` remains Unif bake/UCC floor, not the M−U gap twin.

**Decode (both arms):** `hierarchical_arpc` (T=1) and `hierarchical_arpc_t01`
(T=0.1). Packing: dual-stream `single_stream=false`, `sub_block=null`, s32.

| Row | Profile | Masked `2020048` | Unif `2020049` | Δ(M−U) |
|-----|---------|-----------------:|---------------:|-------:|
| T=1.0 | `hierarchical_arpc` | **55.6** (done) | **33.4** (done) | **22.2** |
| T=0.1 | `hierarchical_arpc_t01` | TBD | TBD | TBD |

### What is the same (procedure parallel)

- Profile pins, packing, thr=null, T, eval set.
- Hard-twin train preset: `xfer_bg_mix_32_blockgen` — only FP arm differs
  (absorbing vs uniform + matching prior/posterior). Config diff per pair
  must show noise kernel only (plus arm label / simplex).
- Procedure: propose → score → reset bad tokens to the **arm’s own** noise.

### What is still different (must not overclaim)

1. **Post-reset state:** MASK vs random word — intentional arm difference.
   Gap measures MASK vs Unif under a shared corrector, not identical
   model-visible conditions.
2. **Scorer identity (locked):** **not** a frozen external AR. Both arms score
   with the **same diffusion ckpt’s backbone** via
   `backbone.causal_logits` → `ar_metric`/`nll`
   (`BlockSampler._ar_log_probs_block`). Matches BlockGen (diffusion model as
   AR predictor). Hard twins trained with size-1 CE + `x0_causal`
   (`bg_weights_1_32` @ 5/95). **C3 skipped ARPC** (no size-1 mixture) — that
   caveat does **not** apply to these hard twins.
3. **Work per arm:** scheduled `num_to_corrupt = round((1−α_s)·L)` is shared,
   but realized reset set / rounds / NFE (incl. AR scoring forwards) can
   differ. **Require in results:** mean reset fraction, guided rounds, mean
   `n_forwards` / NFE per arm (`last_nfe_stats`; enable `track_revisions` when
   logging reset rates). Unequal compute → note, don’t bury.
4. **Same recipe ≠ equally tuned:** floor bump Unif 33.4→38.1 shows hard-U is
   under-tuned vs hard-M near its best — hence the two-row table.

**Do not use as the T=0.1 baseline row:** `hierarchical_quiet` (also flips
`single_stream_decode=true`) or old parity quiet `*_s8` (hard-M quiet **28.4**,
hard-U quiet **1.4** at 8 steps — wrong steps + packing). D3 ~52.5 on `2092151`
is quiet+ss on the floor ckpt, not this table.

## Tier-2 open-loop T×pack 2×2 (no ARPC) — preregistered 2026-10-01

Isolates temperature vs packing. **Same posterior as BlockGen ancestral, not
end-to-end BlockGen.** Unif(V) stays out (Unif-only ablation).

| | Pack as-is (`ss=F`, `sub=null`) | ss+sub8 |
|--|--|--|
| **T=1.0** | `hierarchical_ancestral` (unusable @T=1: ~2.4 / ~5.4) | `hierarchical_ss_ancestral` |
| **T=0.1** | `hierarchical_ancestral_t01` | `ss_quiet_ancestral` |

Submit: `./scripts/submit_openloop_t_pack_matrix.sh` (also queues
`hierarchical_arpc_t01` first). Hard twins only.

| Result @ T=0.1 no ARPC | Read |
|--|--|
| Both arms ≥ ~20% GSM | Collapse was mostly T; rewrite quiet-T story |
| Masked climbs, Unif stays low | Real arm gap in open loop (missing eraser) |
| Both stay ≲10% | Open-loop collapse holds |


**Secondary columns only (headline decode, not matched):** DualCache, UCC/D2,
conf remask, ancestral, `A_ar_sft`.

**To fill the table (cluster-dependent):** (1) D3 SUMMARY on floor for bake
continuity; (2) `hierarchical_arpc_t01` s32 on **both** `2020048` and `2020049`;
(3) keep ARPC knobs identical; (4) optional later: train masked twin of
`2092151` if gap claims should sit on the floor recipe.

Every experiment (anti-N2C, UCC, C3) reports Δ vs these rows.

**Gap diagnosis (M−U under ARPC):** cutoffs + Tests 1–3 + six pre-flights in
[`GAP_DIAGNOSIS_M_U_2026-10-01.md`](GAP_DIAGNOSIS_M_U_2026-10-01.md).
Gap cutoffs use paired CI bounds (≤5 pp = CI upper). Submit Test 1 with Test 3.
Scripts: `tools/gap_test3_denoiser_quality.py` (forward) →
`tools/gap_test2_prefix_oracle.py` (prefix curve) →
`scripts/submit_openloop_t_pack_matrix.sh` (T×pack / `arpc_t01`).

## Decode-method inventory (one row = one run)

**Citation rule:** a row is citeable only if it has a **stated checkpoint**, a
**job ID** (or documented bake job), and a **`SUMMARY.json`**. Log-only GSM
(e.g. D3 ~52.5) is **provisional**. “Best GSM” across configs is selection —
do not collapse multiple ckpts into one cell.

**NFE:** mean NFE (incl. AR-scoring passes) is required for mechanism claims;
most historical rows lack `nfe_metrics.json` → mark NFE as `—` until re-scored.

### Profile knobs (resolved config)

| Code id | Alias | T | thr | ss | sub8 | sticky | cache | ARPC | greedy |
|---------|-------|--:|----:|:--:|:----:|:------:|:-----:|:----:|:------:|
| `hierarchical_arpc` | ARPC T=1 | 1.0 | null | ✗ | — | ✗ | ✗ | ✓ | ✗ |
| `hierarchical_arpc_t01` | ARPC T=0.1 matched | 0.1 | null | ✗ | — | ✗ | ✗ | ✓ | ✗ |
| `hierarchical_ancestral` | ancestral T=1 | 1.0 | null | ✗ | — | ✗ | ✗ | ✗ | ✗ |
| `hierarchical_ancestral_t01` | ancestral T=0.1 | 0.1 | null | ✗ | — | ✗ | ✗ | ✗ | ✗ |
| `hierarchical_quiet` | ARPC T=0.1 **+ ss** | 0.1 | null | ✓ | — | ✗ | ✗ | ✓ | ✗ |
| `hierarchical` / `baseline` | conf remask B1 | 1.0 | 0.9 | ✗ | — | ✗ | ✗ | ✗ | ✓ |
| `hierarchical_ss` | conf remask + ss | 1.0 | 0.9 | ✓ | — | ✗ | ✗ | ✗ | ✓ |
| `hierarchical_ss_ancestral` | ss+sub8 + ancestral T=1 | 1.0 | null | ✓ | 8 | ✗ | ✗ | ✗ | ✗ |
| `hubmatch` | ss+sub8 thr=0.9 | 1.0 | 0.9 | ✓ | 8 | ✗ | ✗ | ✗ | ✓ |
| `dual_cache` | MASK DualCache | 1.0 | 1.0 | ✓ | 8 | ✗ | ✓ | ✗ | ✓ |
| `uniform_commit` | `ucc_thr0.9_b1pack` | 1.0 | 0.9 | ✗ | — | ✓ | ✗ | ✗ | ✓ |
| `uniform_commit_t1` | `ucc_thr1_b1pack` | 1.0 | 1.0 | ✗ | — | ✓ | ✗ | ✗ | ✓ |
| `uniform_commit_ss` | `ucc_thr0.9_sub8` | 1.0 | 0.9 | ✓ | 8 | ✓ | ✗ | ✗ | ✓ |
| `uniform_dual` | **`ucc_thr1_sub8`** | 1.0 | 1.0 | ✓ | 8 | ✓ | ✗ | ✗ | ✓ |
| `uniform_dual_random` | ucc random order | 1.0 | 1.0 | ✓ | 8 | ✓ | ✗ | ✗ | ✓ |
| `ucc_l2r_sub8` | ucc L→R | 1.0 | 1.0 | ✓ | 8 | ✓ | ✗ | ✗ | ✓ |
| `ss_quiet_ancestral` | ss+sub8 T=0.1 anc. | 0.1 | null | ✓ | 8 | ✗ | ✗ | ✗ | ✗ |
| `full_seq_dual` | full-seq escape | 1.0 | null | ✗ | — | ✗ | ✗ | ✗ | ✗ |

**Effective kernel (per arm):** On **masked**, `hierarchical` = conf remask. On
**Unif**, `unmask_threshold` without sticky is a **dead pin** → ancestral unless
UCC sticky is on. Coerced bake rows (`*.COERCED_UCC`) ran as `uniform_commit`,
not hierarchical — do not cite as B1/B3.

**Masked twin:** name the counterpart profile or `none`.

### A. Matched-meter rows (pinned)

| Code id | Arm | Ckpt | Role | Effective kernel | Twin | T | thr | ss/sub | cache | GSM | IFE | NFE | Job | SUMMARY | Status |
|---------|-----|------|------|------------------|------|--:|----:|-------|:-----:|----:|----:|:---:|-----|:-------:|--------|
| `hierarchical_arpc` | M | `2020048` | matched meter | BlockGen ARPC re-MASK | `2020049` same profile | 1.0 | null | ✗/— | ✗ | **55.6** | 22.4 | — | `2104834` / `2074483` | ✓ | **done** |
| `hierarchical_arpc` | U | `2020049` | matched meter | BlockGen ARPC re-Unif | `2020048` same profile | 1.0 | null | ✗/— | ✗ | **33.4** | 19.2 | — | `2104833` / `2074484` | ✓ | **done** |
| `hierarchical_arpc_t01` | M | `2020048` | matched meter | ARPC T=0.1, same pack | U same | 0.1 | null | ✗/— | ✗ | — | — | — | **`2129192`** | ✗ | **queued** |
| `hierarchical_arpc_t01` | U | `2020049` | matched meter | ARPC T=0.1, same pack | M same | 0.1 | null | ✗/— | ✗ | — | — | — | **`2129193`** | ✗ | **queued** |
| `hierarchical_arpc` | U | `2092151` | floor meter (not M−U twin) | ARPC re-Unif on ss+shift floor | none (no M `U0_ss_shift`) | 1.0 | null | ✗/— | ✗ | **38.1** | 19.0 | — | `2096661` | ✓ | **done** (floor, not gap twin) |
| `hierarchical_ancestral` | M | `2020048` | forensic / 2×2 T1 pack | ancestral | U same | 1.0 | null | ✗/— | ✗ | 2.4 | 16.1 | — | `2074480` | ✓ | done |
| `hierarchical_ancestral` | U | `2020049` | forensic / 2×2 T1 pack | ancestral | M same | 1.0 | null | ✗/— | ✗ | 5.4 | 16.8 | — | `2074481` | ✓ | done |
| `hierarchical_ancestral_t01` | M | `2020048` | 2×2 T0.1 pack | ancestral T=0.1 | U same | 0.1 | null | ✗/— | ✗ | — | — | — | **`2129196`** | ✗ | **queued** |
| `hierarchical_ancestral_t01` | U | `2020049` | 2×2 T0.1 pack | ancestral T=0.1 | M same | 0.1 | null | ✗/— | ✗ | — | — | — | **`2129197`** | ✗ | **queued** |
| `hierarchical_ss_ancestral` | M | `2020048` | 2×2 T1 ss+sub8 | ancestral | U same | 1.0 | null | ✓/8 | ✗ | — | — | — | **`2129194`** | ✗ | **queued** |
| `hierarchical_ss_ancestral` | U | `2020049` | 2×2 T1 ss+sub8 | ancestral | M same | 1.0 | null | ✓/8 | ✗ | — | — | — | **`2129195`** | ✗ | **queued** |
| `ss_quiet_ancestral` | M | `2020048` | 2×2 T0.1 ss+sub8 | ancestral | U same | 0.1 | null | ✓/8 | ✗ | — | — | — | **`2129198`** | ✗ | **queued** |
| `ss_quiet_ancestral` | U | `2020049` | 2×2 T0.1 ss+sub8 | ancestral | M same | 0.1 | null | ✓/8 | ✗ | — | — | — | **`2129199`** | ✗ | **queued** |
| `hierarchical_ancestral` | U | `2092151` | forensic | ancestral | none | 1.0 | null | ✗/— | ✗ | 4.9 | 18.7 | — | `2096660` | ✓ | done |

### B. UCC 2×2 on floor `2092151` (decode-only; fill gaps)

D1 vs D2 changes **thr and packing together** — not attributable. Complete the grid:

| | B1 pack (`ss=false`, `sub=null`) | ss+sub8 |
|--|--|--|
| thr=0.9 | `uniform_commit` **D1 35.5** (`2125453`) | `uniform_commit_ss` **never run** |
| thr=1.0 | `uniform_commit_t1` **no clean SUMMARY** | `uniform_dual` **D2 60.1** (`2125454`) |

| Code id | Arm | Ckpt | Role | Effective kernel | Twin | T | thr | ss/sub | GSM | IFE | NFE | Job | SUMMARY | Status / flag |
|---------|-----|------|------|------------------|------|--:|----:|-------|----:|----:|:---:|-----|:-------:|---------------|
| `uniform_commit` | U | `2092151` | control / UCC cell | UCC sticky thr=0.9 B1 | none (Unif-only) | 1.0 | 0.9 | ✗/— | **35.5** | 18.1 | — | `2125453` | ✓ | done |
| `uniform_dual` | U | `2092151` | headline / UCC cell | **`ucc_thr1_sub8`** force-max≈/step | none | 1.0 | 1.0 | ✓/8 | **60.1** | 20.0 | — | `2125454` | ✓ | done; mechanism pending |
| `uniform_commit_ss` | U | `2092151` | control / UCC cell | UCC thr=0.9 + ss+sub8 | none | 1.0 | 0.9 | ✓/8 | — | — | — | — | ✗ | **never run** (priority 3) |
| `uniform_commit_t1` | U | `2092151` | control / UCC cell | UCC thr=1 B1 | none | 1.0 | 1.0 | ✗/— | — | — | — | — | ✗ | **never run** clean (priority 3) |
| `uniform_dual_random` | U | `2092151` | control | same as D2, random site | none | 1.0 | 1.0 | ✓/8 | — | — | — | `2127929` | ✗ | **queued** D5 |
| `ucc_l2r_sub8` | U | `2092151` | control | same as D2, L→R order | none | 1.0 | 1.0 | ✓/8 | — | — | — | `2128010` | ✗ | **queued** L2R |
| `ss_quiet_ancestral` | U | `2092151` | control | ss+sub8 T=0.1 ancestral | **none** | 0.1 | null | ✓/8 | — | — | — | `2127927` | ✗ | **queued** D4 |
| `hierarchical_quiet` | U | `2092151` | secondary | ARPC T=0.1 **+ ss** | M quiet (diff ckpt/steps) | 0.1 | null | ✓/— | ~52.5 | — | — | `2125055`→`2127926` | ✗ | **provisional** (log); r2 queued |

### C. Secondary / not matched (headline + history)

| Code id | Arm | Ckpt | Role | Effective kernel | Twin | T | thr | ss/sub | cache | GSM | IFE | Job | SUMMARY | Status / flag |
|---------|-----|------|------|------------------|------|--:|----:|-------|:-----:|----:|----:|-----|:-------:|---------------|
| `dual_cache` | M | `1762534` | headline | MASK DualCache K/V | **none** (MASK-only) | 1.0 | 1.0 | ✓/8 | ✓ | **64.5** | 22.2 | bake DC | ✓ | done |
| `dual_cache` | U | any | — | — | — | — | — | — | — | — | — | — | ✗ | **failed/OOD** (deleted twin) |
| `hubmatch` | M | `1762534` | headline | ss+sub8 thr=0.9 | U hubmatch (weak) | 1.0 | 0.9 | ✓/8 | ✗ | **62.1** | 20.5 | m2048 | ✓ | done |
| `hubmatch` | M | `2020048` | secondary | ss+sub8 thr=0.9 | U | 1.0 | 0.9 | ✓/8 | ✗ | 43.3 | 11.3 | m2048 | ✓ | done |
| `hierarchical` | M | `1762534` | headline | conf remask B1 | U (see coerced) | 1.0 | 0.9 | ✗/— | ✗ | **56.2** | 22.0 | `2008723` | ✓ | done |
| `hierarchical_ss` | M | `1762534` | headline | conf remask + ss | U coerced | 1.0 | 0.9 | ✓/— | ✗ | **56.3** | 21.8 | `2008724` | ✓ | done |
| `hierarchical_arpc` | M | `1762534` | matched-ish (old C0) | ARPC | old U0 | 1.0 | null | ✗/— | ✗ | **55.6** | 20.3 | bake B2 | ✓ | done |
| `uniform_dual` | U | `1955203` | headline | ucc_thr1_sub8 hubclean pins | none | 1.0 | 1.0 | ✓/8 | ✗ | **64.0** | 21.1 | UDual hubclean | ✓ | done; **not floor-comparable** |
| `uniform_commit` | U | `1955203` | headline | ucc thr=0.9 hubclean | none | 1.0 | 0.9 | ✗/— | ✗ | **57.7** | 21.6 | UC hubclean | ✓ | done; **not floor-comparable** |
| `uniform_dual` | U | `2020049` | control | ucc_thr1_sub8 on hard-U | none | 1.0 | 1.0 | ✓/8 | ✗ | **46.9** | 13.7 | attack | ✓ | done |
| `uniform_dual` | U | `1955203` | forensic | early sticky thr=1 | none | 1.0 | 1.0 | ✓/8 | ✗ | 31.7 | — | hist | ✓ | done (old pin) |
| `hierarchical_arpc` | U | `1955203` | matched-ish (old U0) | ARPC | old C0 | 1.0 | null | ✗/— | ✗ | **25.9** | 20.1 | bake B2 | ✓ | done |
| `uniform_commit` | U | `1955203` | control | bake UC B1pack | none | 1.0 | 0.9 | ✗/— | ✗ | 26.9 | 18.1 | `2001987` | ✓ | done (pre-hubclean) |
| `hierarchical` | U | `1955203` | unusable as B1 | **coerced → UC** | M hierarchical | 1.0 | 0.9* | ✗/— | ✗ | 26.9 | 18.1 | `2008725` | ✓ | **unusable** (COERCED_UCC) |
| `hierarchical_ss` | U | `1955203` | unusable as B3 | **coerced → UC** | M hierarchical_ss | 1.0 | 0.9* | ✓/— | ✗ | 26.9 | 18.1 | `2008726` | ✓ | **unusable** (COERCED_UCC) |
| `hierarchical` | U | `1955203` | forensic | thr dead pin → ancestral | M | 1.0 | 0.9† | ✗/— | ✗ | ~5 | 19.8 | bake B1 raw | ✓ | done (ancestral-like) |
| `hierarchical` | U | `2020049` | forensic | thr dead pin → ancestral | M | 1.0 | 0.9† | ✗/— | ✗ | 1.8 | 12.9 | m2048 | ✓ | done |
| `hierarchical` | U | `2092151` | forensic | thr dead pin → ancestral | none | 1.0 | 0.9† | ✗/— | ✗ | 4.9 | 18.7 | m2048 | ✓ | done |
| `hierarchical_quiet` | M | `2020048` | unusable | quiet **s8** + ss | U quiet s8 | 0.1 | null | ✓/— | ✗ | 28.4 | 12.0 | `2020052` | ✓ | **unusable** (wrong steps) |
| `hierarchical_quiet` | U | `2020049` | unusable | quiet **s8** + ss | M quiet s8 | 0.1 | null | ✓/— | ✗ | 1.4 | 10.0 | `2020053` | ✓ | **unusable** (wrong steps) |
| `hierarchical_quiet` | U | `1955203` | unusable | quiet **s8** | — | 0.1 | null | ✓/— | ✗ | 8.9 | 13.1 | `2011842` | ✓ | **unusable** (wrong steps) |
| `hierarchical_ss_ancestral` | M/U | — | escape | ss+sub8 + ancestral | each other | 1.0 | null | ✓/8 | ✗ | — | — | — | ✗ | see Tier-2 queued (hard twins) |
| `ss_quiet_ancestral` | U | `2092151` | control | ss+sub8 T=0.1 ancestral | **none** | 0.1 | null | ✓/8 | — | — | — | `2127927` | ✗ | **queued** D4 (floor; ≠ hard-twin 2×2) |
| `full_seq_dual` | — | — | escape hatch | full-seq dual | — | 1.0 | null | ✗/— | ✗ | — | — | — | ✗ | **never run** (low pri; C3≠this) |

† On Unif without sticky, thr is ignored (ancestral). \* Coerced rows: label lied; kernel was UCC.

### Priority for never-ran / incomplete

1. `hierarchical_arpc_t01` on `2020048` **and** `2020049` (matched baseline second row) — **queued** `openloop_tpack`.
2. Tier-2 open-loop T×pack 2×2 on hard twins (same batch).
3. D4 / D5 / `ucc_l2r_sub8` (+ D3 SUMMARY) — UCC mechanism controls; need `nfe_metrics`.
4. `uniform_commit_ss` + clean `uniform_commit_t1` on `2092151` — finish UCC 2×2.
5. `full_seq_dual` only if a question needs it.

---

**Ckpts:**  
- Old floors: C0 `1762534` · U0 `1955203` (no size‑1)  
- Soft size‑1: masked `2024324` · uniform `2024327` (`xfer_bg_mix_32`)  
- Hard+causal: masked `2020048` · uniform `2020049` (`xfer_bg_mix_32_blockgen`)  
- SS pack floor: U0 `1994131` (`U0_ss_pack`) · shift-only U0 `1857278` (`U0_shift`)

**Meter:** full suite — GSM `exact_match,flexible-extract` · IFE
`prompt_level_strict_acc` · MMLU (`mmlu` / `mmlu_generative`) · GenPPL+H.
**Primary profiles:** `hierarchical_ancestral` · `hierarchical_arpc` (s32).  
Not quiet-only; not UCC-as-floor. Control: `uniform_dual` on hard‑U only.

---

## Why UCC 58–64% is not enough

On old U0, published-style decodes stay weak (ancestral ~5%, ARPC ~26%).  
Only Hub-clean UCC jumps. Until ancestral/ARPC rise on a size‑1 train, we have
not matched BlockGen’s “uniform works under ancestral/ARPC” story.

Matched soft/hard size‑1 still shows **~22–28pp ARPC M−U** (vs BlockGen
absorb≈uniform ~1–2pp). Treat as pipeline failure, not TinyGSM/data excuse.

---

## Attack matrix (submitted)

| Axis | Preset / lever | Hypothesis | Status |
|------|----------------|------------|--------|
| **N2C close** | `U0_ss_pack` `1994131` / `xfer_bg_mix_32_blockgen_ss` **`2092154`** | Dual-stream GT N2C ≠ open-loop | **DONE** anc 3.3 / ARPC **28.0** (miss) |
| **+ Unifusion shift** | `U0_ss_shift` **`2092151`** / `xfer_bg_mix_32_blockgen_ss_shift` **`2092155`** | shift-as-x0 + ss packing | **DONE** — **floor** 38.1 / ss_shift 35.8 |
| **Shift alone** | `U0_shift` `1857278` | Isolate shift without packing | done |
| **Unif(V)** | `xfer_bg_mix_32_blockgen_unifv` resume **`2094996`** (run `2092156`) | Literal Unif(V) vs `Unif(V\E)` | **DONE** anc 3.4 / ARPC **30.8** (miss gate) |
| **UCC control** | `uniform_dual` on hard‑U `2020049` **`2092157`** | Sticky commit still lifts? | **DONE** GSM **46.9%** / IFE 13.7% |

**AR→full-seq diffusion floor (C3):** Unifusion-style (`block_size=2048`,
uniform+shift) — train resubmit **`2104831`** (Priority). “Full-seq” = diffusion only.  
**Matched causal AR** (Tab-1, not full-seq): `A_ar_sft` = `ar_sft_1857323` —
GSM **70.8** / IFE **24.6** / MMLU **48.4**.  
Harvest: [`HARVEST_ATTACK_C3_2026-09-27.md`](HARVEST_ATTACK_C3_2026-09-27.md).

Default remains `uniform_simplex_mode=conversion`. `blockgen` is ablation only.

---

## GSM8K (%) — cite carefully

### Old U0 / C0 (no size‑1) — bake-off

| Decode | Steps | Masked C0 | Uniform U0 |
|--------|------:|----------:|-----------:|
| Ancestral | 32 | ~2 | ~5 |
| ARPC | 32 | **55.6** | **25.9** |
| Remask / bake UC (`uniform_commit`) | 32 | **56.2** | **26.9** |
| DualCache | 32 | **64.5** | — (MASK-only) |
| Hub-clean UCC / UDual (parity, not bake floor) | 32 | — | **57.7 / 64.0** — different pins than bake UC; do **not** swap for 26.9 |

### Size‑1 soft / hard — ancestral + ARPC (decision meters)

| Train | Arm | Ancestral | ARPC s32 | Δ(M−U) ARPC | Note |
|-------|-----|----------:|---------:|------------:|------|
| Soft | masked `2024324` | **1.9** | **46.8** | | |
| Soft | uniform `2024327` | **5.8** | **18.7** | **28.1** | worse ARPC than old U0 25.9 |
| Hard | masked `2020048` | **2.4** | **55.6** | | matches old C0 ARPC |
| Hard | uniform `2020049` | **5.4** | **33.4** | **22.2** | provisional Unif floor (superseded) |
| Hard | uniform `2020049` | — | — | | `uniform_dual` control **46.9** GSM |

### Attack GSM — decision meters (ancestral + ARPC s32)

| Preset | Job | Ancestral | ARPC | vs hard-U 33.4 | Gate |
|--------|-----|----------:|-----:|---------------:|------|
| **`U0_ss_shift`** | `2092151` | 4.9 | **38.1** | **+4.7** | **PASS → `AR2B-U-floor`** |
| `ss_shift` | `2092155` | 3.0 | **35.8** | +2.4 | PASS (not max) |
| `unifv` | `2094996`/`2092156` | 3.4 | **30.8** | −2.6 | miss |
| `blockgen_ss` | `2092154` | 3.3 | **28.0** | −5.4 | miss |
| hard-U | `2020049` | 5.4 | **33.4** | — | provisional |

**Verdict:** Rule §§1–5 fires → **`AR2B-U-floor = U0_ss_shift`**. Unif(V) hurts vs
`V_eff`. Ancestral still ~3–5% (forensic). Full-suite / hygiene + C3 evals **done**
(C3 collapsed). Next: Layer C on this floor; no new xfer mashups without a protocol bump.
**As of 2026-09-30:** queue empty.

### Decode bake on floor `2092151` (2026-09-30 → 2026-10-01) — Tier 1

Preregistered keep: GSM ≥ **41.1** (floor ARPC 38.1 + 3pp). Else drop vs keep ARPC.

| ID | Profile | Job | GSM | Notes |
|----|---------|-----|----:|-------|
| D1 | `uniform_commit` | `2125453` | **35.5** | Below ARPC; not a win |
| D2 | `uniform_dual` | `2125454` | **60.1** | Clear win vs ARPC; schedule-sensitive |
| D3 | `hierarchical_quiet` | `2125055` → **`2127926`** | ~**52.5** log | Failed exit 15, no SUMMARY — provisional; r2 queued |
| D4 | `ss_quiet_ancestral` | **`2127927`** | — | D2 packing + T=0.1 ancestral; if ≈D2 → UCC mostly T |
| D5 | `uniform_dual_random` | **`2127929`** | — | D2 pins + random force-max site |

**Honesty:** D3 52.5 / IFE 17.7 from failed-job log until resubmit SUMMARY.  
D2 is **`ucc_thr1_sub8`** (legacy id `uniform_dual`): thr=1 ≈ **one force-max commit/step**,
argmax, high NFE — not a MASK DualCache twin. D1 vs D2 may be mostly NFE; plot GSM vs
mean NFE. Do not claim “UCC beats ARPC” without D4/D5 (and ideally L→R) at matched compute.
See [`UNIFORM_CONFIDENCE_COMMIT_2026-09-24.md`](UNIFORM_CONFIDENCE_COMMIT_2026-09-24.md).
