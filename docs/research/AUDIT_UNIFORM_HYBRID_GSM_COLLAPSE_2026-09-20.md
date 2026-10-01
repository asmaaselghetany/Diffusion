# Audit: collapsed Instruct GSM on uniform/hybrid (2026-09-20)

Question: is ~1–7% GSM normal for our uniform/hybrid conversion cells, or a systematic bug?

Generated: 2026-09-20T18:24:25 · refreshed with GSM forensics

**Companion:** `AUDIT_U0_UNIFORM_GSM_2026-09-20.md` (single-cell deep dive) · JSON dump: `AUDIT_UNIFORM_HYBRID_GSM_COLLAPSE_2026-09-20.json`

---

## Verdict

**Systematic generative failure across the whole uniform conversion family — not one bad job.**

| Layer | What we see |
|-------|-------------|
| Train | Healthy: val/ppl ~3.9–4.3 (near C0 3.93) |
| GenPPL+H | Often citeable and *better* than C0 (U0 H̄≈6.0 vs C0 4.80) |
| Instruct GSM | **0.8–7.1% on every uniform cell**; hybrid B4 **16.1%**; C0 control **63.7%** |
| Failure mode | Fluent / half-fluent CoT that never reaches a correct `####` answer — not empty EOS collapse |

This is **not** explained by missing SUMMARY files, one bad seed, or a scorer bug. Flexible-extract still finds digits; accuracy stays near floor.

---

## Scoreboard (best SUMMARY on disk)

| Cell | Job | Corr | val/ppl | GenPPL (H̄) | GSM | IFE | Eval profile |
|------|-----|------|--------:|------------:|----:|----:|--------------|
| C0 masked baseline | `1762534` | masked | 3.93 | 128 (4.80) | **63.7** | 22.2 | dual_cache thr=1 g=1 |
| U0 uniform baseline | `1849335` | uniform | 4.04 | 61 (6.04) | **6.1** | 18.1 | baseline thr=None g=1 |
| U0+shift | `1857278` | uniform | 4.27 | 506 (6.32) | **0.8** | 13.1 | baseline thr=None g=1 |
| U0+anneal | `1857471` | uniform | 4.04 | 54 (6.01) | **6.9** | 17.9 | baseline thr=None g=1 |
| U0 from C0 | `1857519` | uniform | 4.08 | 41 (6.04) | **7.1** | 17.0 | baseline thr=None g=1 |
| xfer block-mix U | `1849287` | uniform | 3.96 | 55 (6.25) | **6.1** | 17.2 | baseline thr=None g=1 |
| U2 joint AR | `1848844` | uniform | 4.20 | 87 (6.26) | **4.5** | 17.4 | baseline thr=None g=1 |
| xfer bg-mix U | `1855541` | uniform | — | 65 (6.45) | **6.7** | 18.3 | baseline thr=None g=1 |
| hybrid B4 anneals | `1857475` | hybrid | 3.94 | 106 (4.57) | **16.1** | 14.8 | baseline thr=None g=1 |
| hybrid p=0.1 | `1836811` | hybrid | 5.52 | 402‡ | — | — | **no Instruct lm-eval** |

‡ uncitable prefix-EOS GenPPL

---

## Train knobs (Hydra) — recipes differ, GSM does not

| Cell | forward | steps | shift | joint_ar | anneal | loss_w | pretrained |
|------|---------|------:|------:|---------:|-------:|--------|------------|
| C0 | masked | 6000 | F | 0 | — | elbo | T |
| U0 | uniform | 6000 | F | 0 | — | None | T |
| U0+shift | uniform | 6000 | **T** | 0 | 0 | None | T |
| U0+anneal | uniform | 6000 | F | 0 | **2000** | None | T |
| U0 from C0 | uniform | **7500** | F | 0 | 0 | None | T |
| xfer block-mix | uniform | 6000 | F | 0 | — | None | T |
| U2 joint | uniform | 6000 | F | **0.3** | — | None | T |
| hybrid B4 | hybrid | 6000 | F | 0 | **2000** | None | T |

Same block_size=32, gbs=256, ignore_bos=True across the board. Knobs move GenPPL a lot; GSM stays floor-level for every pure uniform.

---

## Generation forensics

### U0 dedicated GSM (`lm_eval_gsm8k_fast`, job `1849335`)

- **1319** long answers · mean len ~136 · **`####` count = 0** · strict-match exact = **0.0** · flexible-extract = **6.1%**
- Starts like math CoT, then stalls / corrupts algebra; almost never closes with a final numeric answer line
- Example (games won): *“Let's let's represent the number of games lost as the L \\. … write the equation \( L + 8 = 22.”* — fluent opener, broken latex, wrong equation, no `####`
- Sampler logged **2640** `greedy=true is invalid for uniform → ancestral` warnings (forced ancestral)

### Other uniforms (`paper_gen_cap` suites)

- Same pattern at scale: thousands of long answers, near-zero `####`, word-salad / topic-confused CoT mixed with short A/B/C/D dumps from MMLU-gen
- **34048** greedy→ancestral warnings per cell (force_greedy=1 on paper_gen; sampler overrides for uniform)
- U0+shift (`1857278`) is worst: GSM **0.8%**, GenPPL uncitable 506, densest word-salad

### Hybrid B4 (`1857475`)

- GSM **16.1%** — better than uniforms, still ~4× below C0
- No uniform-greedy override warnings (hybrid path)
- Still fails to emit `####` on audited long answers

---

## Cross-cutting findings

1. **Not empty collapse.** Answers are English; flexible-extract finds digits; accuracy ~0–7%. Distinct from GenPPL≈402 prefix-EOS failure.
2. **Uniform greedy is illegal.** `argmax(q_xs)` would lock prior noise → sampler forces ancestral. Protocol asymmetry vs C0 DualCache/hubmatch thr=1 greedy is real — but word-salad / unfinished CoT remains a uniform generative failure under ancestral too.
3. **Val/ppl ≈ C0** ⇒ conversion fits the NLL objective; Instruct multi-step math does **not** transfer under uniform reverse.
4. **IFE ~13–18%** while GSM dies — short instruction following ≠ multi-step math generation.
5. **GenPPL+H disagree with GSM on purpose.** Uniforms can look “healthier” on free-gen entropy while being useless on GSM.
6. **Recipe-independent.** Baseline / shift / anneal / continue-FT / block-mix xfer / joint U2 → same GSM floor ⇒ **family issue**, not one bad seed.
7. **hybrid_p10 `1836811`** never received Instruct lm-eval (gap).

---

## What is *not* the explanation

- Missing or corrupt `SUMMARY.json` (scores present and consistent across suites)
- Train divergence (val/ppl healthy)
- Single eval timeout / cancelled job (multiple completed packs agree)
- Scorer can’t parse answers (flexible-extract fires; strict `####` almost never present)

---

## Falsifiers / next probes

1. C0 GSM with `DECODE_PROFILE=baseline` ancestral (fair decode match to U0).
2. U0 NFE×GSM steps 32/64/128 + `log_samples` (in flight: `1912772+` / `1912794+`).
3. Side-by-side 20-prompt sheet: C0 dual vs U0 ancestral.
4. Instruct eval for hybrid `1836811`.
5. Keep citing GenPPL+H separately from GSM.

---

## Bottom line for slides

> Uniform AR→block conversion **trains cleanly** and can look strong on GenPPL+H, but **Instruct generative GSM is systematically broken (~1–7%)** across every uniform recipe we ran. Masked C0 stays ~64%. Hybrid is intermediate (~16%). Treat this as a **corruption-family generative failure**, not a flaky job.
