# Gap diagnosis M−U under ARPC — protocol 2026-10-01

**Honest verdict (2026-10-01):** the Unif **denoiser looks mostly healthy** so
far. What is unhealthy is **end-to-end** performance (33–38% vs 55.6% masked
under ARPC). The gap is **not explained** yet. Do not start a training campaign
until Tests 1+2+3 are read together.

**Hard twins (locked):** masked `2020048` ∥ uniform `2020049`
(`xfer_bg_mix_32_blockgen`). Floor `2092151` is **not** the gap twin.

**Noise band:** ≈2 pp on per-arm GSM before calling a single-arm difference
real. For **gaps**, use bootstrap CIs on the paired (M−U) differences, not
only on each arm. A gap of ~5 pp at n≈1300 is within noise — apply the
“≤5 pp” exposure cutoff to the **CI upper bound** of the gap, not the point
estimate. Same idea for “≥15 pp”: use the CI lower bound.

### Cause board (ruling out one at a time)

| Cause | Status |
|-------|--------|
| Sampler kernel wrong | **Ruled out** (matches BlockGen posterior; H2/H2b/H3) |
| Leak / bug in UCC decode | Probes pass; live harness checks pending |
| Probe / code-version problem | **Mostly closed** (utils restored, committed, hashed, submit guard). Val-NLL reproduction **pending** |
| Unif can't denoise | **Not supported** — corrupt-site acc ≈ masked at n=64 |
| Unif overwrites correct tokens | **Not supported at T=1** (soft keep ≈0.99). T=0.1 **open** |
| Temperature | **Untested** on matched pair (`arpc_t01` not run) |
| Exposure (own-error prefix) | **Untested** — Test 2 |
| Training recipe (no time-cond, N2C, V_eff) | **Open** |

### Is Unif healthy?

- **Fine so far:** denoises corrupt sites ≈ masked; random-corruption AUROC ≈0.997; sampler soft-keeps clean tokens at T=1.
- **Real problems:** no identity-copy at α=1.0 (partly OOD vs train α_max≈0.999); **no time conditioning**; large E2E gap while BlockGen's M−U gap is small.
- **Blind spot:** tests use **random** corruption (easy). Own decode-time errors are untested — best suspect until Test 2 speaks.

### ARPC∥ARPC baseline lock (parallel enough)

Same profile is necessary but not sufficient. Before citing the gap as locked:

| Check | Status / rule |
|-------|----------------|
| Config diff (pair) | Only noise kernel + arm label / simplex differ under `xfer_bg_mix_32_blockgen` |
| Scorer | **Same diffusion backbone** `causal_logits` + `ar_metric`/`nll` both arms — not frozen external AR; hard twins have size-1 (unlike C3) |
| Compute | Log reset fraction, guided rounds, mean NFE/`n_forwards` per arm; unequal work → note |
| Two-row report | **Matched twin** 55.6 vs 33.4 (Δ22.2) · **Tuned-Unif reference** 55.6 vs 38.1 (Δ≈17.5, not matched recipe) |

Post-reset MASK vs Unif word is the measured arm difference (procedure-parallel,
not state-identical). See DESIGN_LOCK `ARPC-GAP-BASELINE`.

---

## What's established

- Gap appears under shared ARPC (55.6 vs 33.4 on hard twins).
- BlockGen from-scratch shows only a small M>U gap under ARPC → 20 pp is not
  intrinsic to uniform diffusion; conversion/recipe is a candidate.
- Unif is recipe- and T-sensitive (33.4 → 38.1 floor; quiet ~+14 pp provisional)
  — part of Δ22.2 is tuning, hence the tuned-Unif reference row.
- Masked ARPC at T=0.1 has **never** run on matched packing → cause 1 open.
- Mid-α reverse **kernel** matches BlockGen (line-match + H3 masked; H2 on
  BlockGen simplex `V_eff=V`). A kernel bug is **unlikely** as the M−U cause.
  Conversion simplex `V_eff=V\E` (table default) is covered by **H2b**
  (fast vs naive/materialized TV≈0.01 on small cases) in
  `audit_blockgen_decode_parity.py` — not the same as BlockGen identity, but
  the adaptive algebra is consistent.

### Kernel / collapse verdict (locked wording)

| Claim | Status |
|-------|--------|
| Mid-α kernel matches BlockGen | **Supported** (line-match + H3; H2 for BlockGen simplex) |
| Kernel explains the M−U gap | **Unlikely** |
| Ancestral collapse is just knobs | **Untested** until T=0.1 + ss cells land |
| Gap from ckpt / training surface | **Open** — main suspects: AR→block L=2048 vs 1+32 scratch, and **no time conditioning** |

**Do not overclaim from ancestral T=1 alone.** Hard-twin ancestral
`hierarchical_ancestral` is **unusable at T=1** (**2.4% M / 5.4% U**) — not a
permanent open-loop verdict until `hierarchical_ancestral_t01` lands. Remask
**~56% on the same M weights** shows **masked** needs a corrector; Unif story
open. Floor Unif ancestral **4.9%** is `2092151` (different ckpt). Matched
usable floor for the gap baseline is **ARPC**, not UCC (homemade; controls pending).

## Candidate causes

| # | Cause | Predicts |
|---|--------|----------|
| 1 | Temperature mismatch | Gap shrinks a lot at T=0.1 on **both** arms |
| 2 | Unif harder from AR init | Matched-noise denoise loss/acc worse for Unif |
| 3 | Exposure (N2C) | Unif fine with GT prefix; drops much more on own prefix |
| 4 | Recipe / noise mismatch | Gains from recipe alone (already saw floor bump) |
| 5 | Harness / packing bug | BlockGen TinyGSM ckpt fails to reproduce in our ARPC |
| **6** | **No time conditioning (Unif)** | Unif worse at **low noise** (high α): weak detection; masked nearly time-agnostic via MASK density |
| ~~7~~ | ~~Unif overwrites kept tokens (T=1)~~ | ~~Low clean top-1 ⇒ ancestral trashes good tokens~~ — **not supported at T=1** (see below) |

### Cause 6 — confirmed training fact

**Training forward does not receive `t` / `σ`.** `BlockTrainer` requires
`algo.time_conditioning=False`; `_backbone_logits` always passes `sigma=None`;
Qwen `forward` deletes `sigma`. Per-block `t` is used only to build `xt` and
ELBO weights — never as a backbone input. (Earlier “one scalar t per block”
into the model was wrong for this stack.)

For **masked**, that is often fine: MASK fraction reveals noise level. For
**Unif**, a corrupted token looks like a real word, so the denoiser must infer
how noisy the block is — harder. **Provisional** contributor; Test 3 so far
does not show a large M−U corrupt-site gap.

### Overwrite / retention (Test 3 readout — provisional until n=1000)

| Claim | Status |
|-------|--------|
| Unif overwrites kept tokens under ancestral **T=1** | **Not supported** — clean top-1 is low, but noise-removal **soft keep ≈0.99** (n=8 mini). Keep-xt preserves clean tokens; do not infer overwrite from top-1 alone |
| α=1.0 “no copy” | Partly **OOD** — train α_max≈0.999; trust **0.999 / 0.99** rows |
| Soft keep drops at **T=0.1** (~0.4–0.6 on n=8) vs quiet ARPC gain (~38→52) | **Open** — isolation retention vs ARPC corrector; need n=1000 + retention inside ARPC |
| Test 3 vs Unif denoiser | Corrupt-site M≈U (smoke); retention high at T=1; random-corruption AUROC easy → gap likely **elsewhere** (exposure / T / recipe / ARPC internals) |

**Next most informative after n=1000 + val-NLL:** **Test 2** (prefix-oracle /
exposure). Model-prefix AUROC on *random* replacements is still the easy case —
build detection labels from positions where Unif decode **disagrees with a
correct AR-teacher** solution.

## Preregistered cutoffs (Test 2)

Write before running. Adjust only with a logged protocol bump.
All gap thresholds are on **paired bootstrap CIs**, not point estimates alone.

| Label | Rule |
|-------|------|
| **Exposure-dominated** | Gap@75% **CI upper** ≤ **5 pp** **and** gap@0% **CI lower** ≥ **15 pp** |
| **Denoiser-dominated** | Gap@75% **CI lower** ≥ **15 pp** |
| **Mixed** | Anything between |

## Order

**Login / now (no new training):**

1. Finish **Test 3** n=1000 — read **in-range α** (0.999, 0.99, 0.98) first,
   then retention T=1 vs T=0.1, then train-NLL buckets vs probe clean-site NLL.
2. Reproduce logged **val/nll** with committed code (`tools/reproduce_val_nll.py`).
   Until that lands: forward-process equivalence is **not reproduced**.
3. Build AR-teacher prefix JSONL and run **Test 2** (most likely to explain the
   gap if the denoiser stays comparable).

**When booster returns (decode-only on same ckpts):**

4. `hierarchical_arpc_t01` + open-loop T×pack 2×2 on both twins
   (`scripts/submit_openloop_t_pack_matrix.sh`).

Only after those readouts choose a training change.

Scripts:

1. Test 3 — `tools/gap_test3_denoiser_quality.py`
2. Test 2 — `tools/gap_test2_prefix_oracle.py` (`build-prefixes` then `run`)
3. Val-NLL — `tools/reproduce_val_nll.py`
4. Test 1 submit — `scripts/submit_openloop_t_pack_matrix.sh`

---

## Pre-flight checks (before GPU)

Cheap sanity. A scripting error here is hard to spot after the run.

### Test 3

1. **Corrupted set = `xt ≠ x0`.** Under Unif a redraw can equal the original
   (~1/V or 1/V_eff). Those positions are **not** corrupted — exclude them from
   AUROC labels and corrupt/clean accuracy. Log collision / drop counts.
2. **Script sanity (CPU `self-check`).** Oracle denoiser (returns x0) →
   accuracy 1.0 and AUROC 1.0. Uniform-random predictor → accuracy ~1/V and
   AUROC ~0.5. If either fails, the metric code is wrong, not the model.
3. **Masked AUROC = N/A**, not 1.0. MASK is visible by construction; a trivial
   1.0 invites a misleading table comparison.
4. **α_t matching.** Both arms use the same α → t map after any
   `U0_ss_shift` / schedule bump. Match on **α**, not raw t, if the time-to-α
   map differs.
4b. **Record `time_conditioning_received`.** Must be `false` for both arms
   (cause 6 premise). Script refuses if unexpectedly true.
4c. **Noise-level split.** After the α sweep, read Unif high-α vs low-α
   clean/AUROC (`noise_level_split_uniform.pattern`).
4d. **Probe-check before full run** (`probe-check` subcommand):
    - Oracle/random self-check (clean+corrupt).
    - **α=1.0 row** (no corruption): if model copies, clean_acc≈1; if still
      ~0.25, either no-copy behavior or a probe bug.
    - Fix **`V_eff`** logging (refuse on error). Corrupt := `xt≠x0`; log
      move-mask collisions separately (not keep-rate).
    - Masked clean_acc / AUROC = **N/A** (SUBS / visible MASK).

### Test 2

5. **Anchor at f=0%.** Must reproduce 55.6 and 33.4 (within noise) under
   `hierarchical_arpc` before other f. If not, the new runner differs from the
   old harness — stop.
6. **Fixed problem set.** Same problems across all f and both arms. Drop any
   problem too short to cut at 75% before the answer. Teacher prefix enters
   through the **same** `prefix_ids` → `sampler.generate(...)` path on both
   arms so the clean-stream half is identical.

Run: `python tools/gap_test3_denoiser_quality.py self-check`

---

## Test 3 — denoiser at matched α_t

**Q:** Can each model denoise? Can Unif detect wrong tokens?

**Setup**

- Ckpts: `2020048` / `2020049`.
- Data: held-out conversion batches + optional GSM-style sequences; **same**
  examples and block grid for both arms.
- Clean stream = **GT** (N2C condition) so exposure is off.
- Corrupt **one** active block (prefer block index ≥ 1 so a GT previous block
  exists). Match on **α_t** (fraction left clean):
  `{1.0, 0.999, 0.99, 0.98, 0.95, 0.90, 0.70, 0.50, 0.30, 0.10}`
  → `t=(1-α)/(1-eps)`. **α=1.0** is the copy probe (slightly above train
  α_max≈1−ε); **0.999/0.99** are in-range. Log train α range in JSON.
- Masked → MASK; Unif → random from train simplex (`V_eff`). Corrupt :=
  `xt ≠ x0`. Log move-mask collisions separately.

**Metrics** (per arm × α): corrupt acc/CE; Unif clean top-1 (**masked = N/A**);
Unif soft clean stats (`clean_ce`, mean `p(x0)`, mean `p(xt)`, rank of x0);
Unif detection AUROC `1-p(xt)` (**masked = N/A**).
**Retention:** one real Unif ancestral noise-removal step at α∈{0.99,0.95,0.90},
T∈{1.0,0.1} — fraction of clean tokens kept (`keep_xt` / `still_x0`).
**Model-prefix:** prev block = model fill; re-eval active block (decode-time
AUROC). ≥1000 blocks/level; paired gap CIs.

**Record in the JSON (required):** `time_conditioning_received: false` (from
ckpt / `BlockTrainer` contract) — yes/no answer to cause 6’s premise.

**Noise-level split (cause 6 probe):** compare Unif at high α (low noise,
e.g. 0.95/0.90) vs low α (high noise, e.g. 0.30/0.10). Pattern that supports
missing time-cond: **clean-site accuracy and/or detection AUROC worse at high α**
(model must know most tokens are right, but can’t read noise off the input).
Flat Unif metrics across α → model ignores noise level entirely (same conclusion,
different symptom).

**Optional (cheap, decode):** run Unif once with a deliberately wrong assumed
α-schedule in the reverse posterior; flat GSM vs correct schedule → score does
not use noise level. Not required before Test 3 lands.

**Read first when the n=1000 table lands:** in-range α rows (0.999, 0.99, 0.98);
then retention T=1 vs T=0.1; then train-NLL buckets vs probe clean-site NLL
(settles whether soft-NLL discrepancy was a probe artifact). Treat n=8 mini
magnitudes as **illustrative only** until then.

### `forward_process/utils.py` restore (2026-10-01)

Committed `HEAD` **had** a **45-line stub** (never contained `normalize_uniform_*` /
`V_eff` helpers — only ever in the working tree until today's commit). At ~13:44
the worktree file was truncated to that stub; restored ~13:46 from `stash@{0}`
(`pre-authorship-rewrite`).

| Artifact | `utils.py` |
|----------|------------|
| SHA256 | `1a3f6c494ed8bee8ea63fb66d7ff2b3e3f00cb3bf13555a3cb877749d1202f9e` (276 lines) |
| vs stash | **Identical** |
| vs pre-restore HEAD | Differs — was the 45-line stub |
| Smoke `smoke_login_n64` (13:24) / `probe_check` (13:33) | **Before** truncate → full worktree code |
| Failed n=1000 restart | Hit truncated file (ImportError) — discarded |
| Current n=1000 + post-restore `self-check` | Restored full (= stash) |

**Train-era proof (partial):** job `2020049` (2026-09-26 03:40–15:09) used
`submit_dir=Diffusion-new`, `PYTHONPATH=…/Diffusion-new/src`, and successfully
instantiated `BlockUniformForwardProcess` (imports `normalize_uniform_simplex_mode`).
Wandb saved **no** code artifact; current file mtime is the restore, not train
start; train logs do **not** print `V_eff`. So: train **had** the helper API;
byte-identity with today's stash is **not** proven. A different `V_eff` /
exclude set / `normalize_uniform_*` would still import. Closing test:
reproduce logged `val/nll` with today's code (`tools/reproduce_val_nll.py`).

> Forward-process code at train time is not recoverable byte-for-byte; the
> full API was present (checkpoint loads the dependent class), and validation
> NLL is **not reproduced** yet with the committed version (pending login-GPU
> run after n=1000). Do not imply functional equivalence until that lands.

See `out/gap_test3/UTILS_RESTORE_CHECK.md`. n=1000 JSON has no fingerprint
field — use sidecar `out/gap_test3/full_login_n1000.CODE_VERSION.md`.

**Masked arm vs `utils.py`:** masked training/eval imports only
`_mask_token_id` / `_effective_vocab_size` from the same file (via
`block_masked.py`). It does **not** call `normalize_uniform_*` / `V_eff`. The
45-line stub still provides those two helpers, so a stub checkout can train
**masked** and only break on **Unif**. Hash guard + fingerprint cover both
arms' shared file.

**Mitigations now:** full `utils.py` committed + baseline tag; every eval JSON
gets `code_fingerprint.forward_process_utils.sha256` (startup assert refuses the
stub); submit scripts (`_block_qwen_env`, `_common`, `submit_lever`,
`submit_family_eval`, openloop matrix) refuse a hash mismatch before sbatch.
Check booster / clean-clone copies — `Diffusion-codex-fixes` still has the stub.


## Test 2 — prefix-oracle curve

**Q:** Exposure vs weak denoiser?

**Setup**

1. Prefixes from **correct** `A_ar_sft` solutions (not dataset gold style).
2. Cut at a **block boundary before the final answer**;
   f ∈ `{0%, 25%, 50%, 75%}` of solution tokens (aligned to block_size).
   Drop problems that cannot support a real 75% cut — fixed mix for all f.
3. Both arms continue under `hierarchical_arpc` and `hierarchical_arpc_t01`
   via the same `prefix_ids` path.
4. Same problems, seeds, extractor; log mean NFE incl. AR-scoring passes.
5. Report GSM(f) per arm and Δ(M−U)(f) **with gap CIs**. Run f=0% first
   (anchor to 55.6 / 33.4).

**Pitfalls:** equal cut tokens across arms; no answer in prefix; style check
AR prefixes vs Instruct; config diff only the noise kernel; same n and CIs.

## Joint readout

| Observation | Meaning | Fix |
|-------------|---------|-----|
| Gap large at f=0%, small at 75%; Test 3 gaps small | Exposure | Anti-N2C / scheduled sampling |
| Gap stays large at 75%; Test 3 Unif worse | Weak denoiser / recipe | Longer train, schedule, hybrid noise |
| Acc fine, Unif detection AUROC low at ARPC noise | Can't find errors | Error-detecting objective (GIDD/SCDD) |
| Unif worse at high α than low α (clean / AUROC) | Missing time-cond hurts Unif | Add σ/t (or noise-revealing train signal) |
| Unif flat across α | Denoiser ignores noise level | Same fix family as time-cond |
| Gap flat in f | Not exposure | Recipe, harness, T, or ARPC internals |
| Soft keep ≪1 at T=1 on clean sites | Overwrite under ancestral | Sticky / copy objective — **not seen on n=8 mini** |
| Soft keep drops at T=0.1 but quiet ARPC rises | Retention vs corrector puzzle | Retention inside ARPC; n=1000 |
| Mostly T (Test 1) | Quiet T default; gap smaller than thought | Update baseline rows |

## Scripts / submits

| Piece | Path |
|-------|------|
| Test 3 | `tools/gap_test3_denoiser_quality.py` |
| Test 2 | `tools/gap_test2_prefix_oracle.py` |
| Test 1 submit | `scripts/submit_openloop_t_pack_matrix.sh` |
| Inventory / baseline | [`AR2BLOCK_GSM_TABLE.md`](AR2BLOCK_GSM_TABLE.md) |
