# Experiments log

Each subdirectory is one run or diagnostic. **Every experiment gets an `EXPERIMENT.md`**
with design, jobs, diagnostics, problems, and verdict.

**Daily default (from 2026-09-06):** each calendar day of scientific work also gets
`experiments/YYYY-MM-DD_<slug>/EXPERIMENT.md` written at **thesis depth** (motivation,
method, evidence, rejected alternatives, job ledger, open questions, draft thesis
bullets). See `.cursor/rules/daily-experiment-logs.mdc`. Index every new day below.

## Layout

```
experiments/
  README.md                          ← this file
  <YYYY-MM-DD>_<name>_<jobid>/
    EXPERIMENT.md                    ← required
  <YYYY-MM-DD>_<theme>/              ← day log (multi-job / diagnosis OK)
    EXPERIMENT.md                    ← required, thesis-usable detail
    artifacts/ or artifacts → links  ← optional copies / notes
```

## Index

| Folder | Type | Status |
|--------|------|--------|
| [2026-08-22_ar2block_uniform_1457696](2026-08-22_ar2block_uniform_1457696/EXPERIMENT.md) | Train + eval (Pipeline 1, uniform) | Train OK; soup; **gen-PPL 72.3** |
| [2026-08-23_ar2block_masked_1467698](2026-08-23_ar2block_masked_1467698/EXPERIMENT.md) | Train + eval (Pipeline 1, masked) | Train OK; soup; **gen-PPL 266.1** |
| [2026-08-24_block_masked_1467699](2026-08-24_block_masked_1467699/EXPERIMENT.md) | Train + eval (Pipeline 2, masked) | Train OK; soup; **gen-PPL 375.1** |
| [2026-08-24_block_uniform_1467700](2026-08-24_block_uniform_1467700/EXPERIMENT.md) | Train + eval (Pipeline 2, uniform) | Train OK; soup; **gen-PPL 384.3** |
| [2026-08-25_collapse_control](2026-08-25_collapse_control/EXPERIMENT.md) | Diagnostic: AR Instruct vs block-on-AR-init | **Done** — AR fluent; AR-init block = soup (sampler implicated) |
| [2026-08-25_ar2block_uniform_arpc_decode](2026-08-25_ar2block_uniform_arpc_decode/EXPERIMENT.md) | Transfer-shaped ARPC decode on ar2block | **Blocked** — need size-1 mixture; native ARPC is `B3_arpc` / `blockgen_*` on `line=block` |
| [2026-08-25_genppl_rerun](2026-08-25_genppl_rerun/EXPERIMENT.md) | Re-run gen-PPL after gpt2-large prefetch | **Done** (1491192–1491195) |
| [2026-08-31_postfix](2026-08-31_postfix/RUN_PROTOCOL.md) | Post-fix protocol (families + phases) | Living protocol |
| [2026-09-04_reference_recreation](2026-09-04_reference_recreation/EXPERIMENT.md) | Fast-dLLM + BlockGen recreate | Living; collapse notes → 2026-09-06 |
| [2026-09-06_collapse_plain_ce_eval](2026-09-06_collapse_plain_ce_eval/EXPERIMENT.md) | Day log: C2 plain_ce sign bug, C0 diagnosis, multi-node lm-eval | **Fix landed**; train **1694554**; C0 lm-eval **1694730** |
| [2026-09-07_lm_eval_nccl_timeout](2026-09-07_lm_eval_nccl_timeout/EXPERIMENT.md) | Postmortem + gap-close: family eval router, ARPC pins, DualCache lm-eval | C0 **1700915**; fastdllm exactness **1700927** |

### Four-arm gen-PPL snapshot (gpt2-large, first_chunk_only)

| Arm | gen-PPL |
|-----|---------|
| ar2block_uniform | **72.3** |
| ar2block_masked | 266.1 |
| block_masked | 375.1 |
| block_uniform | 384.3 |

## Conventions

1. **Do not overwrite** an existing experiment folder when re-running; create a new dated folder and link back.
2. Link to Slurm job IDs, `outputs/block_qwen/...`, and `slurm_logs/...`.
3. Record **problems** even when training “succeeded” (e.g. soup samples with good ELBO).
4. Tag every run with **family**: conversion (`ar2block`) | native (`block`) | transfer (`xfer_*`). Report **`line × arm × data × budget`**. Never call `ar2block`+BlockGen knobs a BlockGen recreate.
5. Prefer compute-visible `/e/` paths in all docs.
6. **Day logs** should be long enough to draft thesis Experiments / Limitations without re-opening chat transcripts: include rejected alternatives, metric definitions, and hedged “thesis notes” bullets.

## Shared infra notes (2026-08-25)

- Dataset cache race fixed (`fcntl` lock in `loaders.py`); 4-node jobs load prebuilt `.dat`.
- `generate_samples()` uses `training.sampling_eps` when `eps=None` (matches train/decode).
- DepBench not installed on this workspace; eval soft-skips instead of hard-failing.
- `gpt2-large` required for gen-PPL; prefetch via `scripts/download_block_qwen_data.sh`.
- Collapse-control uniqueness heuristic misses high-entropy soup; `latin_ratio` soupish check added 2026-08-25.
