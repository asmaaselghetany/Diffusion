# Deep iterate audit — abnormal U0 behaviour (2026-09-22)

## Smoking guns found and fixed

### 1. GSM `max_new` silently 512 while PROTOCOL said 2048 (REAL)

Canary log: `generate_until … max_new=512` with `MAX_NEW_TOKENS=2048` in
`PROTOCOL_LOCK.json`. Cause: `min(task.max_gen_toks, adapter_max)` — lm-eval
gsm8k defaults to 512 and defeated Hub budget.

**Fix:** `block_qwen_lm_eval.py` uses adapter `max_new_tokens` then
`cap_max_new_for_task` only. Pending GSM jobs will pick this up on start.

### 2. ChatML / PAD in Unif *noise* simplex (REAL, Instruct-odd)

Qwen: `im_start`, PAD/`endoftext` sat in `Unif(V\{MASK})` while MASK did not.
Masked never injects those as noise; uniform could redraw `<|im_start|>` /
PAD into CoT during train and reverse.

**Fix:** Unif noise / prior / ARPC redraw / limiting / ELBO `V_eff` now exclude
`{MASK, PAD, im_start}`. EOS/`im_end` stays allowable so the model can stop
via `p(x0)`.

## Ruled out again (not “secretly broken”)

- Duo ELBO algebra / `xt|x0` attention flags / shift-off on U0
- Prompt wipe / future Unif under hierarchical
- MASK leakage in canary gens (0 MASK strings; fluent wrong algebra)

## Still abnormal vs C0 — expected residuals (not code bugs)

| Item | Why it still looks “wrong” |
|------|----------------------------|
| Ancestral Unif vs DualCache thr=1 | Different reverse physics |
| Old ckpt `Unif(V)` vs new exclusions | Needs retrain `1952660` for lock |
| No TinyGSM 1+32 / T=0.1+ARPC on canary | Quiet/ss probes pending |
| Fluent wrong CoT | Capability / recipe; hygiene ≠ Instruct |

## Pending jobs (will see max_new + decode exclude; train exclude on retrain)

1952657 C0 ancestral · 1952658 U0 ss · 1952736 quiet · 1952660 retrain ·
1953129/30 hygiene
