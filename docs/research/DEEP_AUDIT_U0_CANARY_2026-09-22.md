# Deep audit: U0 hierarchical canary still ~6% GSM (2026-09-22)

**Job:** `1950105` COMPLETED · **ckpt:** `1849335` · **profile:** hierarchical / greedy=0  
**Out:** `…/ar2block_uniform_1849335/lm_eval_hier_gsm_canary_20260922/`

Companion falsified: `HARD_AUDIT_UNIFORM_GSM5_FORENSIC_2026-09-21.md` P0
(“full-seq future Unif is why GSM≈6%”).

---

## Verdict

**Something is still extremely wrong with U0 Instruct generation — but it is
not the protocol pin we just fixed.**

| Check | Result |
|-------|--------|
| Hierarchical engaged? | **Yes** (`PROTOCOL_LOCK`, `tok_s` `hierarchical_kv=true`, ~13 tok/s vs ~2.4 baseline) |
| Greedy forced off? | **Yes** (`force_greedy=0`) |
| ChatML / MASK leaks? | **Clean** (0 `<\|im_*\|>`, 0 MASK strings) |
| GSM flexible | **6.37%** (±0.67) — same band as baseline 6.06–6.60% |
| GSM strict | **0%** |
| Sample quality | English-fluent CoT, **broken algebra**, `\boxed{}` not `####` |
| “Correct” flex (84/1319) | Mostly **lucky digit scrape** from garbled LaTeX |

**Hierarchy alone does not move U0 off the ~6% scrape floor.** Do not requeue
the U-family on that hope.

---

## What the canary rules out

1. Silent demotion to `hierarchical_kv=false`
2. Greedy locking prior Unif tokens
3. ChatML continuation / MASK token leaks / scorer-only failure  
   (C0 ~64% on the same flexible path)
4. “Just pin hierarchical and GSM jumps toward C0”

Consistent with `DEEP_AUDIT_UNIFORM_ATTENTION_2026-09-22.md`: under dual-stream
block attention, future keys past the active block are already blocked — so
truncating with hierarchical was never a ~6%→60% switch.

---

## Cross-cell evidence (all ancestral / broken-or-fixed protocol)

Every uniform Instruct GSM we have sits in the **4–7%** band (except
`U0_shift` **0.8%** under the old baseline+greedy paper_gen — untrustworthy):

| Cell | GSM flex (best known) |
|------|------------------------|
| U0 `1849335` baseline | 6.1–6.6% |
| U0 `1849335` **hierarchical canary** | **6.4%** |
| U0_from_C0 `1857519` | ~7.1% |
| U0_anneal `1857471` | ~6.9% |
| xfer `1849287` / mix_u `1855541` | ~6.1–6.7% |
| U2 `1848844` | ~4.5% |

Masked C0 `1762534` hubmatch/dual_cache: **~63%**. Same mix/budget/block.

---

## Ranked remaining problems

### P0 — Residual failure mode (real)

**Fluent wrong reasoning under ancestral uniform reverse**, not geometry demotion.

- Starts like GSM CoT, then invents equations (`$77` for `$7`, `-10=-1x`, …).
- Flexible ~6% ≈ **84 lucky extracts / 1319**.
- Train looks “healthy”: val NLL≈1.40, GenPPL first_chunk citeable — **hygiene ≠ Instruct**.

This is the thesis-critical U0 failure. Decode protocol locks held; capability did not appear.

### P1 — Why it can still look “broken” next (ordered)

1. **Unfair C0 comparison**  
   C0 GSM uses `dual_cache` + `threshold=1` + greedy confidence. U0 **cannot**
   use that reverse (uniform). Never ran **C0 baseline ancestral** as control.
   If C0 ancestral also collapses, part of the gap is decode recipe unfairness.

2. **Ancestral uniform error accumulation**  
   Within-block redraw keeps positions editable; early wrong tokens poison later
   blocks. BlockGen TinyGSM often uses **low T + ARPC**. We have not run that
   on U0 under locked hierarchical.

3. **Ckpt train contract drift**  
   `1849335` trained under older `Unif(V)` + ELBO `V`; current code is
   `Unif(V\{MASK})` + `V_eff`. One-token simplex drift — **not** enough alone
   for 6% vs 63%, but **retrain** required before claiming locked uniform.

4. **Packing residual vs BlockGen**  
   We decode dual-stream `concat(xt,x0)` with `x0=xt` on the active block;
   BlockGen feeds `[clean prefix | active xt]` single-stream. Attention audit
   says xt does **not** attend same-block x0 (only prior x0 blocks) — so this
   is P1/P2, not a fresh canary smoking gun. Still worth a
   `single_stream_decode=true` smoke.

5. **`parameterization: subs` on uniform yaml**  
   BlockGen uniform uses `parameterization: mean`. Our trainer **forces**
   `subs` in code but `_uniform_loss` skips SUBS and uses DUO NLL — label is
   misleading, likely not the GSM killer. Clean up on retrain.

### P2 — Accepted / secondary

- Init `top1_agreement≈3.5%` (AR vs block graph) — expected; masked recovers.
- GSM `max_new` capped at 512 while PROTOCOL says 2048 — secondary.
- Strict 0% from missing `####` — **same as C0**; not U0-specific.
- `U0_shift` 0.8% under old baseline+greedy — do not cite.

---

## Sample forensics (canary)

| Signal | Value |
|--------|-------|
| n (flex filter) | 1319 |
| `####` in gens | ~5–10 |
| `\boxed` in gens | ~977 |
| flex exact_match | 84 |
| MASK / ChatML leaks | 0 |

Even “wins” are often garbled `\boxed{…}` that flexible-extract lucks into.

---

## Recommended next moves (do not spray queue)

1. **Fairness control (cheap):** C0 `1762534` GSM under **ancestral** profile
   (no DualCache / no thr) — same harness as U0 canary.  
   - If C0 stays high → gap is corruption/recipe.  
   - If C0 tanks → we overstated U0 doom vs C0.

2. **Decode probe (cheap):** U0 hierarchical + **ARPC** + lower temperature /
   quieter ancestral (BlockGen-style), `single_stream_decode=1` smoke.

3. **Train lock (expensive):** Retrain U0 under current `V_eff` + MASK-free
   FP, then hierarchical eval. Optional: `U0_shift` under locked protocol
   (old 0.8% invalid).

4. **Do not** requeue full U-family solely because hierarchical is “fixed.”

---

## Bottom line

The canary did its job: **protocol is fixed; U0 Instruct GSM is still ~chance-level
fluent garbage.** The remaining bug is in **uniform conversion learning and/or
ancestral uniform decode science**, not another `DECODE_PROFILE` pin.
