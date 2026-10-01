# Smoking gun: uniform + greedy = completely wrong decode

**Date:** 2026-09-23  
**Ckpt:** U0 `1849335` (lever-off baseline)

## Claim

The “baseline is multilingual soup” observation is **not** “uniform ancestral is hopeless.”
It is **`greedy=1` on the uniform reverse**, which is mathematically illegal.

## Evidence (same ckpt, same `DECODE_PROFILE=baseline`, same full-seq geometry)

| Run | `greedy` | Sample signature | Weird-char rate | Score |
|---|---|---|---|---|
| `lm_eval_ifeval_samples_baseline` (Sep 17) | **1** | multilingual token soup | ~14% | IFE ~10.7* |
| `lm_eval_ifeval_samples_baseline_ancestral` (Sep 18) | **0** | English Instruct babble | ~2.6% | IFE **18.1** |
| `lm_eval_hier_ss_gsm_20260922` | **0** + hierarchical_ss | fluent wrong CoT, `\boxed{}` | — | GSM flex **6.8%** |

Soup log init args (verbatim):

```text
decode_profile=baseline, greedy=1
```

Ancestral log:

```text
decode_profile=baseline, greedy=0
```

Sep 17 log has **zero** `invalid for uniform` warnings → that run executed **before**
`BlockSampler.generate` auto-disabled greedy on uniform. It actually argmax’d `q_xs`.

## Why argmax is broken (Duo / BlockGen algebra)

Uniform posterior puts mass on **keeping `xt`**. At mid-schedule (`α_t≈0.3`),
even when `p(x0)` is a **one-hot on the true token**, `argmax(q)` still prefers
the current noise id:

| `α_t` (intermediate) | greedy lock `xt` | greedy hit true |
|---|---|---|
| 0.05 | ~0% | ~100% |
| **0.3** | **100%** | **~0%** |
| 0.7 | ~0% | ~100% |

Final noise-removal (`α_s=1`) recovers true when `p` is peaked. Mid-schedule
greedy **writes Unif prior into the lattice**; later steps cannot unbake soup.

Ancestral sampling still redraws `x0` / Unif with positive probability → English
tokens survive → fluent-wrong CoT, not soup.

## What this means for “baseline without levers”

1. **Soup = decode completely wrong** (greedy on Unif). Fixed in code:
   - `BlockSampler.generate` forces `greedy=false` on uniform
   - `block_qwen_lm_eval` forces the same
   - U0 submit scripts default `FORCE_GREEDY=0`
2. **Fair lever-off floor** = ancestral (+ now coerced `hierarchical_ss`):
   fluent wrong Instruct, GSM ~6–9%, IFE ~18%. Bad for math, **not** token soup.
3. Masked C0 ~64% is DualCache thr=1 **commit**, not ancestral. Matching that
   on uniform is `uniform_dual` sticky (queued), not “fix Duo algebra.”

## Do not cite

- U0 IFE 10.7 / soup samples under `greedy=1`
- GenPPL~61 as fluency (hygiene only; open free-gen still wanders)

## One-step denoise recovery (2026-09-23, ChatML text, not random ids)

| Packing | t | move_rate | acc_moved |
|---|---|---|---|
| dual (train graph) | 0.2 | 0.19 | **0.25** |
| dual | 0.5 | 0.48 | **0.134** |
| dual | 0.8 | 0.78 | **0.070** |
| single (decode packing) | 0.5 | 0.48 | **0.122** |

`acc_moved ≫ 1/V` ⇒ weights are not dead. Dual ≈ single ⇒ packing is not the
GSM killer. Mid/high-t recovery is weak → fluent-wrong ancestral GSM (~7%)
is consistent with a soft denoise head, **not** scrambled logits. Soup required
**greedy**.

Artifacts: `outputs/.../eval/one_step_recovery_{dual,single}.json`

## Next proof cells (already queued)

- `uniform_dual` GSM canaries
- `U0_ss_pack` / `U0_ss_shift` retrain (train↔decode packing + V_eff lock)
