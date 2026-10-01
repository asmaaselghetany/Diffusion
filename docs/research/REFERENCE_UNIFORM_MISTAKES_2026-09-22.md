# Reference uniform papers vs our U0 — remaining mistakes (2026-09-22)

Companion to `REFERENCE_UNIFORM_IMPLS_2026-09-21.md` and
`DEEP_AUDIT_U0_CANARY_2026-09-22.md`. Hierarchy canary (~6.4%) falsified
“full-seq Unif attention is why GSM≈6%.” This note re-reads BlockGen /
Duo / Unifusion for what is *still* wrong.

Sources: `third_party/blockgen` ([jdeschena/blockgen](https://github.com/jdeschena/blockgen)),
Duo via BlockGen `DUO_BASE`, Unifusion paper-only, Fast-dLLM (masked only).

---

## What refs never claim

**No reference claims Instruct-scale AR→uniform conversion works.**

| Ref | Actual success surface |
|-----|------------------------|
| BlockGen | TinyGSM (SmolLM-135M, L=512) + OWT/LM1B **scratch** Block-DiT |
| Duo | Full-seq USDM LM (OWT-class) |
| Unifusion | Full-seq GenPPL / shift-as-x0; **not** Instruct GSM |
| Fast-dLLM | Masked conversion only |

U0 Instruct GSM ~4–7% across cells is therefore **not** “we failed to copy a
known working Instruct recipe.” It may still be a real conversion gap.

---

## Ranked remaining mistakes (post-canary)

### P0 — Recipe / capability (not another profile pin)

1. **Fluent wrong CoT under ancestral uniform reverse**  
   Train NLL/GenPPL look healthy; GSM flexible ≈ lucky digit scrape. Same
   floor on U0 / shift / anneal / from-C0 / xfer / U2.

2. **Unfair C0 decode citation**  
   C0 ~63% = DualCache + thr=1 **greedy confidence**. Uniform **cannot**
   use that reverse (argmax locks prior Unif). Need C0 ancestral control
   (queued `1952657`).

### P1 — BlockGen TinyGSM decode science we under-matched

3. **Temperature + ARPC**  
   TinyGSM scripts: best results at **T≈0.1**; headline method is
   **AR-then-ARPC** (`ar_metric`/`nll`), often `STEPS=8` → ~16 NFE/block.  
   Our hierarchical canary: **T=1.0, no ARPC**.  
   Profile `hierarchical_quiet` now pins T=0.1 on both `x0_temperature` and
   `arpc_temperature` + BlockGen ARPC (queued / resubmitted).

4. **ARPC predictor was not BlockGen-identical (fixed 2026-09-22)**  
   BlockGen predictor: `sample_uniform_posterior(..., alpha_s=1,
   noise_removal_step=True)` — keeps `xt` with `keep_xt_prob`.  
   Ours previously: raw `sample_categorical(p(x0))` (always redraw).  
   Fixed in `BlockSampler._arpc_guided_step`.

5. **U0 train recipe ≠ TinyGSM uniform train**  
   BlockGen TinyGSM: **1+32 mixture**, 5% AR / 95% diffusion, **CE@size-1**,
   `pure_noise_block_sizes=[1]`, `x0_causal=True`.  
   Our U0: **fixed bs=32**, empty mixture → ARPC size-1 gate skips unless
   `FORCE_ARPC=1` (decode-only probe; AR verify untrained).

6. **Ckpt `1849335` train≠decode simplex**  
   Trained `Unif(V)`+ELBO`V`; code now `Unif(V\{MASK})`+`V_eff`. Retrain
   queued (`1952660`).

### P2 — Structural residuals (accepted / secondary)

7. **Generate packing**: BlockGen `[clean prefix | active xt]` single
   stream vs our dual-stream `concat(xt,x0)` with `x0[active]=xt`.
   Attention audit: future keys already blocked — not the canary killer.
   `hierarchical_ss` / `single_stream_decode` is the closer pin (queued
   `1952658`).

8. **Per-block IID `t`** vs BlockGen one-`t`-per-seq — shared with working C0.

9. **PAD logit ban at decode**, float64 default off, Hub padded vocab in
   Unif (~0.18%).

### Already matched (do not re-litigate)

- Duo/BlockGen ELBO algebra (+ our `V_eff` adaptation)
- `parameterization: mean`, log-linear `eps=1e-3`, no shift on U0
- Fast uniform posterior, ancestral (no greedy) on uniform
- Train dual-stream block attention (mirrored layout)
- MASK excluded from Unif simplex in current code

---

## TinyGSM script pins (copy targets)

```text
# train (1+32): third_party/blockgen/scripts/train/tinygsm/blockgen_uniform_1_32_w_0p05_0p95.sh
# sample:     .../sample/tinygsm/blockgen_uniform_ar_then_arpc.sh
TEMP≈0.1  STEPS=8  arpc.corruption_mode=ar_metric  arpc.ar_metric=nll
model.x0_causal=True  sampler=ar-then-arpc
```

---

## Probe status (2026-09-22)

| Job | Purpose |
|-----|---------|
| 1952657 | C0 ancestral GSM fairness |
| 1952658 | U0 `hierarchical_ss` |
| 1952659→resubmit | U0 `hierarchical_quiet` after ARPC predictor+T fix |
| 1952660 | U0 retrain under `V_eff` lock |

Do **not** requeue the full U-family hoping hierarchy alone moves GSM.
