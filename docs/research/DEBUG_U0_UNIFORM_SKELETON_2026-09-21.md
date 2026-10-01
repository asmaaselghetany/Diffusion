# Deep debug: why U0 Instruct GSM ≪ C0 (2026-09-21)

**Companion (refs):** `REFERENCE_UNIFORM_IMPLS_2026-09-21.md` — BlockGen / Duo / UDLM / Unifusion mismatch table + full fix contract.

**Claim under test:** U0 should not be ~10× worse than C0 on GSM if the
shared AR→block skeleton is sound — look for bugs, not “uniform is doomed.”

**C0** `1762534` masked · GSM **63.7** (dual_cache/hubmatch)  
**U0** `1849335` uniform · GSM **6.1–6.6** (baseline ancestral, confirmed on
greedy=0 rerun)

---

## Verdict

Several **real skeleton bugs / mismatches** stack on the uniform arm. Training
NLL can still look fine (val/ppl≈4, GenPPL healthy) while Instruct generative
decode is corrupted. Ranked by expected impact:

| # | Finding | Type | Fixes GSM without retrain? |
|---|---------|------|----------------------------|
| 1 | U0 eval used **full-seq** forwards → attention over **future Unif(V) noise**. BlockGen only feeds **clean prefix + active block**. C0 hubmatch/dual uses truncated `active_end`. | Protocol / geometry bug | **Yes** — `DECODE_PROFILE=hierarchical` |
| 2 | `_uniform_step` **skipped** `align_shift_logits` + MASK/PAD ban that `_masked_step` always applies | Sampler bug | **Yes** for U0+shift (was 0.8%); partial for U0 |
| 3 | Prior / block-init / ARPC redraw drew **MASK** as noise; `skip_special_tokens=True` **deletes** those ids → scrambled CoT | Sampler / FP bug | **Yes** (decode); train FP also fixed for new runs |
| 4 | Hybrid uniform branch uses **V\{MASK}**; pure uniform used full V | Train/sample inconsistency | Decode fixed now; full train parity needs retrain |
| 5 | C0 cite numbers are DualCache thr=1 greedy; U0 was bare baseline | Fairness confound | Run C0 `baseline` ancestral as control |

Items 1–3 are enough to treat “6% GSM” as **not** a clean capability measurement.

---

## 1. BlockGen geometry vs our U0 baseline (smoking gun)

BlockGen block view (`third_party/blockgen/samplers.py` `_prepare_block_view`):

- `x0` = **already-decoded prefix only**
- `xt_b` = **current block**
- Model never sees future tokens

Our U0 `DECODE_PROFILE=baseline`:

- `hierarchical_kv=false` → `_use_block_scope()=false`
- `_logits` → full `backbone_logits(xt, x0)` over **entire 2048**
- Future positions are Unif(V) junk (multilingual noise ids)
- Dual-stream attention keys/values for the active window are polluted

C0 paper numbers:

- `hubmatch` / `dual_cache` → truncated `active_end` (no future MASK ocean in the forward)

So the “same skeleton” was **not** the same decode geometry. Masked full-seq is
tolerable (future = single MASK embed); uniform full-seq is hostile (future =
random real tokens).

**Fix shipped:** `submit_family_eval.sh` routes uniform/hybrid generative
lm-eval to **`hierarchical`** (truncated scope, greedy=0, no confidence thr).

---

## 2. Uniform reverse skipped logit prep

`_masked_step` always:

```text
_prepare_masked_logits → align_shift_logits + ban MASK/PAD
```

`_uniform_step` previously:

```text
raw logits → softmax  (no shift, no ban)
```

Consequences:

- **U0+shift** (`1857278`, `shift_loss_targets=true`): train scores token `i`
  from logits `i-1`; sample used unshifted logits → near-random math → GSM **0.8%**
- MASK probability mass stayed in `p_x0` / limiting Unif(V)

**Fix shipped:** `_uniform_step` and ARPC predictor path call
`_prepare_masked_logits`; limiting distribution zeros MASK (`1/(V-1)` elsewhere).

---

## 3. MASK-as-noise + `skip_special_tokens`

lm-eval decode:

```python
tokenizer.decode(cont, skip_special_tokens=True)
```

If any continuation id == `mask_id`, it is **removed**, not shown as a placeholder
→ local word salad / broken equations even when the model was “trying.”

**Fix shipped:** `BlockTrainer.prior_sample`, `BlockSampler._init_block` /
ARPC redraw, and `BlockUniformForwardProcess` redraw MASK hits (Unifusion /
hybrid convention: `Unif(V\\{MASK})`).

---

## 4. What is *not* broken

| Check | Result |
|-------|--------|
| Hydra train match C0 | Same data cache, 6k steps, gbs 256, bs 32, len 2048, hooks off; only `forward_process_name` |
| `loss_weighting` absent on U0 yaml | Trainer defaults → `elbo` (same family as C0) |
| Val / GenPPL | Healthy — does **not** prove Instruct decode is correct |
| Chat template / 0-shot GSM | Matched C0 paper_gen |
| Greedy flag alone | Rerun greedy=0 → same ~6% (not the only bug) |
| Posterior algebra | Matches BlockGen `uniform_posterior_probs` (DUO form) |

---

## 5. Code / config diffs (this session)

- `src/discrete_diffusion/sampling/block_sampler.py`
  - `_uniform_noise`, `_uniform_limiting`
  - `_uniform_step` uses `_prepare_masked_logits`
  - init / ARPC redraw exclude MASK
- `src/discrete_diffusion/algorithms/block_trainer.py` — `prior_sample` excludes MASK
- `src/discrete_diffusion/forward_process/block_uniform.py` — train FP excludes MASK
- `scripts/submit_family_eval.sh` — default uniform profile **`hierarchical`**

---

## 6. Immediate experiments (priority)

1. **Re-eval U0** `1849335` with fixed code + `DECODE_PROFILE=hierarchical`
   `FORCE_GREEDY=0` `SUITE=paper_gen` `OUT_DIR=.../lm_eval_hier_fix_20260921`
2. Same for **U0+shift** (largest expected jump from shift-align fix)
3. **C0 control:** `DECODE_PROFILE=baseline` ancestral GSM (fair floor)
4. If hierarchical still weak: NFE 64/128 on U0 hierarchical
5. Only then consider retrain U0 with MASK-free FP (item 4)

---

## Bottom line

U0 vs C0 was **not** an apples-to-apples skeleton comparison. Full-seq Unif
attention + missing shift/MASK prep on the uniform reverse path are concrete
bugs. Re-measure Instruct GSM after hierarchical + sampler fixes before
declaring a corruption-family failure.
