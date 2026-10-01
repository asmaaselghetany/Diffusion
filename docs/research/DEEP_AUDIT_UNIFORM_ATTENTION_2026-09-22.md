# Deeper uniform pipeline audit — 2026-09-22

## Headline correction (important)

Under our **dual-stream block attention mask**, queries **never attend future
xt / future x0 keys**. Full-seq vs `active_len` truncation are
**attention-equivalent** for allowed KV (verified numerically: remap(full)
== trunc; future key set empty).

So the earlier claim “baseline attends future Unif junk” is **overstated for
this Qwen dual-stream mask**. Hierarchical still matters for:

- shorter SDPA / better float16 numerics with sparse `-inf`
- BlockGen-style packing scope / compute
- not for “suddenly seeing clean future”

**Implication:** hierarchical re-eval alone may **not** lift GSM from ~5% →
C0-like ~63%. The old ~6% run also had MASK-in-noise + missing logit prep +
greedy pin; those decode fixes are still necessary. Expect canary to tell us
whether the model under ancestral uniform is merely weak, or still broken.

---

## What is solid (no cancel-worthy code hole)

| Check | Result |
|-------|--------|
| `active_len` slices tokens out of dual-stream forward | Real |
| Block mask blocks future Unif | Real (train + decode) |
| `_uniform_step` shift-align + MASK/PAD ban | On |
| Fast posterior + `V_eff` + MASK-free redraw | On |
| Protocol refuse: uniform + baseline / greedy=1 | On (sbatch + sh + Python) |
| U0 hydra: `shift_loss_targets=false`, bs=32, steps=32 | Matches hierarchical pins |
| Offline gsm8k / mmlu_gen / ifeval | Cached |

---

## Deep residuals (ranked)

### P1 — can still yield “bad” GSM without being a silent protocol bug

1. **Ancestral uniform ≠ C0 hubmatch confidence** — C0 uses single-stream +
   thr=1 greedy unmask; U0 cannot. Fairness gap remains.
2. **Dual-stream packing ≠ BlockGen `[prefix\|block]` DiT** — accepted residual.
3. **Ckpt `1849335` train contract** — trained before current MASK/`V_eff` lock;
   decode-only claim only.
4. **`ignore_bos=true` zeros loss on pos0 = `<\|im_start\|>`** (Qwen bos is
   `None`). Shared with C0 (so not uniform-specific), but misnamed.
5. **GSM `cap_max_new=512`** while C0 tables used 2048 — can truncate long CoT.

### P2

- PAD banned at decode (`<\|endoftext\|>`); train uniform ELBO only bans MASK.
- EOS = `<\|im_end\|>` — stop_on_eos can work if model emits it.
- first_chunk GenPPL can still look fine while mid-seq soup appears under long
  free-gen (error accumulation).

---

## Ops decision (this pass)

- **Cancelled** 8-node paper_gen ×7 (`1949963`–`1949972`) — too expensive if
  canary stays near ~5%.
- **Kept / resubmitted** U0 GSM canary only: 1 node, `hierarchical`,
  `greedy=0`, `LOG_SAMPLES=1`,
  `…/lm_eval_hier_gsm_canary_20260922`.

When canary finishes, check:

1. `PROTOCOL_LOCK.json` → `hierarchical` / `force_greedy=0`
2. log line `hierarchical_kv=True`
3. sample quality (not multilingual soup; real arithmetic)
4. flexible-extract score

Only then re-queue the full uniform family.
