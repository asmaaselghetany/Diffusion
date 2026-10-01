# Uniform pipeline gate audit — 2026-09-22 (pre-queue)

**Action taken:** cancelled pending jobs `1949656`–`1949663`, hardened
protocol locks, resubmitted as `hier_fix_20260922_v2` + U0 GSM canary.

---

## Why we cancelled

Could not prove Slurm baked `DECODE_PROFILE=hierarchical` into job env
(`scontrol` hides env). `lm_eval.sbatch` still **defaults to baseline** if
unset — the exact failure mode behind U0 GSM ~6%.

Also found: `DECODE_PROFILE=auto` + `blockgen_arpc` previously set
`FORCE_GREEDY=1` and left profile as `auto` (broken).

---

## Locks shipped (must fail before 8-node burn)

| Layer | Guard |
|-------|--------|
| `lm_eval.sbatch` | Infer `EVAL_FORWARD`; refuse uniform/hybrid + `baseline`/`auto`; refuse `FORCE_GREEDY=1`; auto→`hierarchical`+`greedy=0` |
| `lm_eval.sh` | Same refuse |
| `block_qwen_lm_eval.py` | Coerce baseline→hierarchical; re-force `hierarchical_kv`; **RuntimeError** if still full-seq; force `greedy=false` |
| `BlockSampler.generate()` | Re-enable `hierarchical_kv` if demoted after `__init__` |
| `PROTOCOL_LOCK.json` | Written at job start under `OUT_DIR` |

---

## What is still accepted risk (not a protocol bug)

- Old ckpts trained under `Unif(V)` ≠ current `V_eff` train contract — scores are
  **decode-protocol** claims only.
- Dual-stream packing ≠ BlockGen DiT packing (residual).
- Hierarchical may still score poorly if the model is weak — but it will not
  silently re-run the baseline soup protocol.

---

## Offline preflight

`mmlu_generative`, `gsm8k`, `ifeval` load OK under `HF_*_OFFLINE=1`.

---

## Resubmit plan

1. U0 GSM **canary** (1 node) — earliest signal that protocol lock holds.
2. Full **paper_gen** ×7 under `lm_eval_hier_fix_20260922_v2`.
