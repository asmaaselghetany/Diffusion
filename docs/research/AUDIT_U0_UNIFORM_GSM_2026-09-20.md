# Audit: baseline uniform U0 (`1849335`) Instruct GSM ~6%

**Verdict:** Not a scoring glitch and not a failed train. Training/ELBO/GenPPL look healthy. **Generative GSM under uniform ancestral decode produces fluent garbage** (broken algebra / word salad). Same pattern on every other uniform cell (~1–7% GSM). Masked C0 on DualCache+thr=1 is ~62–64% on the same GSM template.

Date: 2026-09-20 · ckpt `ar2block_uniform_1849335/checkpoints/last.ckpt`

---

## 1. What was checked

| Layer | Result |
|-------|--------|
| Train Hydra vs C0 | Matched: 6000 steps, gbs 256, block 32, len 2048, same Nemotron cache, hooks off. Only real diff: `algo=block_uniform` vs `block_masked`. |
| `loss_weighting` | U0 config dumps `null`; trainer defaults null → **`elbo`** (`block_trainer.py`). Same effective loss family as C0. |
| Val metrics (WandB) | U0 val/ppl **4.04** vs C0 **3.93** — not collapsed. |
| ELBO sweep (U0) | bs1 CE ppl 3.76 · bs32 ELBO ppl 5.34 — sane. |
| GenPPL hygiene | first_chunk **60.8 / H̄ 6.04** `low_ppl_healthy_H` — citeable. |
| GSM protocol | 0-shot, chat wrap, `until` includes `<|im_end|>` — **same as C0 paper_gen**. |
| Decode | U0: `baseline`, steps 32, `FORCE_GREEDY=1` → **forced ancestral** (uniform cannot greedy). C0: `dual_cache` thr=1 greedy. |
| GSM score | flexible-extract **6.07%** · strict-match **0%** (no `####`; C0 strict also 0%). |
| Outputs | 1319/1319 non-empty; look like English math CoT but **wrong / garbled**. |

---

## 2. Smoking guns in the generations

From `lm_eval_gsm8k_fast/lm_eval.log` (1320 logged `answer:` dumps):

- Model **does** continue the chat template and writes multi-step “reasoning.”
- Content is **not** EOS-empty collapse (contrast scratch GenPPL≈402).
- Algebra is broken: contradictory equations, duplicated words (“let's let”, “Max's's”), shredded LaTeX.
- Example failure mode (U0): invents `J = 2(J-5)` then simplifies to `0 = -10`.
- Matched C0 on the same Charmaine-age style item: clean `16-12=4` then `+4` — correct structure.

Sampler already documents why greedy is disabled for uniform:

```text
Uniform reverse: argmax(q_xs) preferentially keeps xt (prior noise) when
p_x0 is diffuse — locks multilingual soup into later blocks.
```

Logged **1320×** during GSM:
`sampling.greedy=true is invalid for uniform reverse … Forcing greedy=false (ancestral).`

So GSM ran ancestral BlockSampler — intended for uniform — and still produced soup on math CoT.

---

## 3. Not explained by “wrong profile” alone

| Claim | Check |
|-------|-------|
| Hubmatch wrongly applied? | No — SUMMARY `decode_profile=baseline`, thr=null. Family router forbids hubmatch on uniform. |
| Empty / truncated scoring? | Flexible extract finds numbers in almost every answer; accuracy still ~6%. |
| Few-shot mismatch? | Both U0 and C0 GSM runs are **0-shot**. |
| Chat template mismatch? | Both use `<|im_start|>` chat wrap in logs. |
| Train didn’t finish? | `last.ckpt` = `1-6000.ckpt`; val ppl fine. |
| Only this job? | All uniforms in Fri pack: GSM **0.8–7.1%**. Systematic. |

Protocol asymmetry remains: C0 Instruct numbers are DualCache/hubmatch; U0 is ancestral baseline. That can move numbers, but **does not turn correct CoT into word salad**. IFE on U0 ancestral is **18.1%** (vs C0 ~22%) — instructions partially work; **GSM math does not**.

---

## 4. How to read this for the thesis

1. **Uniform baseline is not “secretly good on GSM.”** Cite 6% with ancestral baseline, or don’t claim Instruct parity.
2. **GenPPL+H and GSM disagree.** U0 looks healthy on collapse hygiene and bad on generative math — exactly why Unifusion-style dual reporting exists.
3. **Corruption is doing real work** on generative Instruct — not just a decode UI choice.
4. Optional fairness probe (if you re-queue): run **C0 with `DECODE_PROFILE=baseline` greedy=false** on GSM only, to quantify DualCache uplift vs corruption. Do **not** run hubmatch/DualCache on U0 as a “fix.”

---

## 5. Recommended follow-ups (priority)

1. **log_samples=1** GSM N=50 for U0 + C0 baseline — side-by-side sheet (not only log truncation).
2. **NFE sweep on GSM** (steps 32/64/128) for U0 — see if soup is under-denoising.
3. **C0 baseline ancestral GSM** — fair decode-matched comparison.
4. Keep uniform GSM in the ledger as **real** until a probe overturns it.

---

## 6. Numbers to keep

| Metric | U0 `1849335` | C0 `1762534` |
|--------|-------------:|-------------:|
| val/ppl | 4.04 | 3.93 |
| GenPPL (H̄) first_chunk | 60.8 (6.04) | 127.9 (4.80) |
| IFE (best) | 18.1 ancestral | ~22 hub/dual |
| GSM flexible | **6.1** baseline ancestral | **62–64** dual/hub |
| GSM strict | 0 | 0 |
