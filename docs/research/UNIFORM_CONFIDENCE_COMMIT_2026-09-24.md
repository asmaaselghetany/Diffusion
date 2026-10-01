# Uniform Confidence Commit (UCC) — 2026-09-24

**Our construction** (not a published named method). **Not DualCache.**

`use_block_cache=false`. UCC only reuses DualCache’s *commit schedule knobs*
(thr / force-max / ss+sub8 packing). It does **not** earn a “masked DualCache
twin” claim: undecided sites stay **Unif tokens**, not MASK, so the model does
not see an explicit undecided lattice.

Paper-faithful Unif floors stay ancestral / ARPC (**B1–B4** /
`AR2B-U-floor`). UCC is a homemade decode probe beside them — cite only with
controls and **GSM vs mean NFE**.

## Inspiration (borrowed pieces, not equivalence)

| Source | Borrowed | What we do **not** get |
|---|---|---|
| Fast-dLLM DualCache | conf-until; thr + force-max; argmax @ T=0 | DualCache K/V; MASK undecided |
| Duo / USDM | Undecided = **Unif**, never MASK | MASK-visible state |
| BlockGen ARPC revise | Optional (`uniform_commit_revise`) | Default **off** (not Hub) |

## Algorithm (per sub-window)

1. Prior = Unif (USDM), not MASK
2. Loop (dt=None): logits → `p_x0` → propose **argmax** → commit if
   `max p_x0 ≥ thr` **or** force-max (peak ≥ `sticky_min_conf`, default 0)
3. Non-commits **keep Unif** (no Duo posterior scribble)
4. Frozen sites never re-Unif
5. Optional revise (off by default)

### Honest thr=1 read (`uniform_dual` / D2)

`max p_x0 = 1.0` almost never happens, so **almost every commit is force-max**:
one highest-conf site per step inside the active window. With `sub_block_size=8`
that is up to **8 forwards per 8-token window** — greedy iterative decode with a
confidence-chosen order, not a multi-token threshold eraser.

So D1 (`thr=0.9`) vs D2 (`thr=1`) is **not** clean “config sensitivity”: D2 spends
far more NFE. Plot **accuracy vs mean NFE** before claiming a mechanism.

Also: commit token is **argmax** (no sampling). Part of any lift vs T=1 ARPC
overlaps the same effect as quiet T / greedy ancestral.

## Correctness before goodness

Homemade is fine if the rule only uses the model's own outputs. Ask **is it
correct (no leak)** before **is it good (beats matched-compute baselines)**.

### Part 1 — leakage / correctness (do first)

| # | Test | Status |
|---|------|--------|
| 1 | Corrupt GT in `x0` / no answer kwargs on `generate` | CPU probes green (`tests/test_decode_leakage.py`) |
| 3 | Future-visibility (mutate beyond `active_end`) | CPU probes green |
| 5 | Clean-stream source = model commits only (UCC uses `block_eval_logits(xt)`) | CPU probes green |
| 4 | Committed-token invariant (UCC + masked remask) | CPU probes green |
| 7 | Slow reference UCC vs fast path | CPU green (`tests/test_ucc_reference.py`); includes force-max **ties** + thr=1 edge |
| — | Sub-window packing: Unif `active_end == window_end` | CPU green (later Unif never densified) |
| 2 | Swapped-prompt live eval | Offline proxy: swapped **6.5%** = adjacent chance **6.5%** (random-pair chance 4.0%) on 200 D2 dumps — at chance, not a leak signal. Live swapped-prompt still required |
| 6 | Extractor / 20 raw outputs by hand | Log dumps OK on D2 (0-shot; reasoning→`\\boxed`); use `audit_decode_leakage.py extractor` |

**Cite-now phrasing:** UCC (`ucc_thr1_sub8`) passes single-stream, committed-invariant,
future-visibility, and slow-reference probes (unit level, stub backbone); no GT
surface found; live harness A/B and matched-NFE mechanism controls pending.

**Wording until then:** valid decode setting, mechanism pending.

Run: `python tests/test_decode_leakage.py` · `python tests/test_ucc_reference.py`.
Live corrupt-GT A/B, swapped-prompt, and batch=1 vs batch=8 invariance need a real
model job — stubs miss bf16/batching/padding. Cluster currently `ReqNodeNotAvail`.

**C3 (`2125372`, `ar2block_uniform_2125372`)** is full-seq: do **not** assume
block-UCC probes transfer. Include it in the same harness job (swapped-prompt,
corrupt-GT, batch invariance + attention/clean-stream rechecks).

### Order once booster is healthy

1. Harness job on **both** `2092151` and C3 `2125372` (live swapped-prompt,
   corrupt-GT, batch=1 vs 8).
2. D4 / D5 / L→R + real `nfe_metrics`.
3. C3 eval set under matched decode profiles.
4. Thr sweep vs mean NFE (once counters trusted).

### Part 2 — is it actually good (after Part 1)

Matched-NFE: D4 / D5 / L→R (queued). Selection on held-out dev. Replicate on
a second ckpt/seed. Confidence–correctness calibration. Non-GSM (IFE).

| Result | Verdict |
|--------|---------|
| Fails any Part 1 | Leak/bug — fix first |
| Passes Part 1, ties D4/L→R at matched NFE | Valid greedy decode; say that |
| Passes Part 1, beats controls, replicates | Real method |
| Passes Part 1, GSM-only / ckpt-fragile | Narrow gain — scope the claim |

## Profiles

Prefer descriptive names in prose. Legacy code ids kept for OUT_DIR continuity.

| Code id | Prefer saying | thr | packing | Notes |
|---------|---------------|-----|---------|--------|
| `uniform_commit` | UCC thr=0.9 (B1 pack) | 0.9 | dual, sub=null | Bake UC; lower NFE |
| `uniform_dual` | **`ucc_thr1_sub8`** | 1.0 | ss + sub8 | D2; ≈ force-max/step |
| `uniform_dual_random` | ucc_thr1_sub8 **random** | 1.0 | ss + sub8 | D5: force-max site random |
| `ucc_l2r_sub8` | ucc_thr1_sub8 **L→R** | 1.0 | ss + sub8 | Fixed L→R one-token/step; matched NFE if D2 is force-max-dominated |
| `ss_quiet_ancestral` | ss+sub8 + T=0.1 ancestral | — | ss + sub8 | D4: no UCC/ARPC |
| `uniform_commit_t1` | UCC thr=1 on B1 pack | 1.0 | dual, sub=null | Isolates thr from ss |
| `uniform_commit_ss` | UCC thr=0.9 + Hub ss | 0.9 | ss + sub8 | |

Coerce never remaps `dual_cache` → UCC (`baseline`→`hierarchical` only).
MASK DualCache twin `uniform_dualcache` **deleted** (OOD).

## Floor bake `2092151` (cite carefully)

| ID | Profile | GSM | Read |
|----|---------|----:|------|
| D1 | ucc thr=0.9 | **35.5** | Below ARPC 38.1 — not a win |
| D2 | ucc_thr1_sub8 | **60.1** | Strong decode setting; **unaudited** mechanism |
| D3 | hierarchical_quiet | ~**52.5** log | Provisional until SUMMARY |
| D4 | ss_quiet_ancestral | queued | If ≈D2 → mostly T/greedy/packing |
| D5 | random force-max | queued | If ≈D2 → confidence **order** unused |
| L2R | ucc_l2r_sub8 | queued | JID `2128010`; trivial order at matched NFE |

NFE/commit-share harvest (CPU): `python tools/harvest_ucc_nfe.py` →
`Diffusion/outputs/block_qwen/ar2block_uniform_2092151/ucc_nfe_harvest.csv`.
D1/D2 lack `nfe_metrics.json` until re-scored with the sampler counters; treat
their rows as accuracy + `tok_s` proxy only. Thr sweep (0.5/0.7/0.9/1.0 on D2
packing) waits until counters are trustworthy.

### Decision rules (all at roughly matched NFE)

| Result | Meaning |
|--------|---------|
| D2 > L→R and D2 > D5 | Confidence ordering matters. Real mechanism claim. |
| D2 ≈ L→R > D5 | Order matters, but trivial L→R ≈ confidence. No eraser story. |
| D2 ≈ D5 ≈ L→R | Order doesn't matter. Gain is greedy + compute, not UCC. |
| D4 ≈ D2 | Temperature + greedy explain it. Drop twin framing. |

If D1 (35.5) and D2 (60.1) land on the **same accuracy–mean-NFE curve** from the
thr sweep, they are two compute points, not a config-sensitivity finding.

**Do not write “UCC beats ARPC” from D2 alone.** Do not call D2 a remask/MASK twin
until D4/D5/L→R settle at matched NFE.

## Failure / fix (2026-09-24)

Job `1995951`: `uniform confidence commit exhausted its bound`.
Fix: budget = remaining unfrozen; on stall flush and continue
(`block_sampler.py` UCC branch).

Wire: `decode_profiles.py` · `_apply_uniform_sticky` · `lm_eval.sh` allowlist.

See also: `DESIGN_LOCKS.md` (`U0-UNIFORM-COMMIT`) · `AR2BLOCK_GSM_TABLE.md` · `BAKEOFF_2026-09-24.md`.
