# Day log — lm-eval NCCL timeout forensics + fastdllm_v2 train exit audit

**Date:** 2026-09-07  
**Site:** Jülich Supercomputing Centre, Jupiter **booster** (GH200)  
**Working tree:** `/e/project1/scifi/elsayed3/Diffusion-new`  
**Family in scope:** **conversion** (`LINE=ar2block`)  
**Parent day:** [2026-09-06_collapse_plain_ce_eval](../2026-09-06_collapse_plain_ce_eval/EXPERIMENT.md)  
**Parent recreate:** [2026-09-04_reference_recreation](../2026-09-04_reference_recreation/EXPERIMENT.md)

Overnight the two “pipeline sanity” jobs both left the queue. This day is a
postmortem: what actually succeeded, what failed, and why the multi-node
lm-eval path aborted despite nearly finishing GSM8K generation.

---

## 1. Where we stood

| Job | Intent | Overnight outcome |
|-----|--------|-------------------|
| `1694554` | Post-`plain_ce` fastdllm_v2 retrain (~1B tok / `max_steps=1900`) | Train reached step 1900 with **positive** loss; Slurm state **FAILED** exit 6 |
| `1698982` | Auto offline eval of `1694554` `last.ckpt` | **COMPLETED** (ELBO + gen-PPL + samples) |
| `1694730` | C0 lm-eval roof, 4×4 Accelerate, partial suite | **FAILED** after ~4h50m; no `SUMMARY` / `results_*.json` |

Queue empty at morning check. No claim of Fast-dLLM recreate success yet.

---

## 2. Research questions

**RQ-A.** Is `1694554`’s FAILED state a real train failure or teardown noise after a
good checkpoint?

**RQ-B.** Why did `1694730` die with NCCL `ALLREDUCE` timeout, and did generation
actually work?

**RQ-C.** What minimal harness change makes multi-node lm-eval viable at ~2 tok/s
BlockSampler speed?

---

## 3. Methods

- `sacct` / `squeue` for terminal states; `slurm_logs/lever_fastdllm_v2_masked_1694554.{out,err}`,
  `slurm_logs/bqwen-lm-C0_1694730.out`.
- Parsed `trainer/loss=` trajectories; inspected
  `outputs/block_qwen/ar2block_masked_1694554/{checkpoints,eval}/`.
- Timeline of `1694730`: first `Running generate_until`, answer-block counts,
  `[tok/s]` write, first `ALLREDUCE` watchdog lines.
- Read lm-eval `evaluator.py` multi-GPU path (`accelerator.gather` →
  `generate_until` → `wait_for_everyone` → `gather_object`).
- Compared ELBO JSON for C0 `1660579` vs fastdllm `1694554`.

---

## 4. Results — fastdllm_v2 train (`1694554`) / offline eval (`1698982`)

### 4.1 Training objective sanity: confirmed

From `slurm_logs/lever_fastdllm_v2_masked_1694554.out`:

- Loss trajectory: **+6.39 → ~1.1** over the run; **no negative losses** in
  ~60k parsed `trainer/loss=` hits (min ≈ 0.60).
- Progress bar reached dataloader step ~30400/42594 while the trainer’s
  `max_steps=1900` fired:  
  `Training finished at step 1900 (>= max_steps=1900); submitting eval for …/ar2block_masked_1694554/checkpoints/last.ckpt`
- Auto-submitted **`1698982`**, which **COMPLETED** in ~40 min.

Checkpoint artifact (valid):  
`/e/project1/scifi/elsayed3/Diffusion-new/outputs/block_qwen/ar2block_masked_1694554/checkpoints/last.ckpt`  
(mtime Sep 6 23:31; also `0-1000.ckpt`, `0-1500.ckpt`, `best.ckpt`).

### 4.2 Why Slurm says FAILED

`sacct`: `1694554` **FAILED** `ExitCode=6:0`. The `.err` ends in
`c10::DistBackendError` / NCCL `ncclRemoteError` on IB during
**process-group teardown after** the “Training finished … submitting eval”
line. Rank-0 aborted (`srun: error: jpbo-008-36: task 0: Aborted`).

Interpretation: **max_steps completion + checkpoint write succeeded**; the
FAILED flag is a **post-success distributed teardown race**, not inverted-loss
collapse and not a missing checkpoint. Treat the run as **scientifically
usable** for train-objective confirmation; do not cite Slurm FAILED as evidence
the recipe is broken.

### 4.3 Offline eval numbers (job `1698982`)

Under `outputs/block_qwen/ar2block_masked_1694554/eval/`:

| Meter | fastdllm `1694554` | C0 `1660579` (prior) |
|-------|--------------------|----------------------|
| gen-PPL (`first_chunk_only`) | **72.2** (`gen_ppl_metrics.json`) | ~146 |
| ELBO block-32 mean NLL / BPD | **2.73 / 3.94** | **1.82 / 2.62** |
| ELBO block-1 BPD | 3.08 | 2.13 |

Nuance: better free-gen head PPL than C0 does **not** imply better conversion;
`1694554` was trained with `plain_ce` (not ELBO weighting), so worse ELBO BPD is
expected-ish and must not be over-interpreted. Chat lm-eval remains the roof
metric. Samples live at `eval/samples.pt` / `samples.txt`.

---

## 5. Results — C0 lm-eval (`1694730`) NCCL abort

### 5.1 What ran

- Launch: 4 nodes × 4 GPUs, `accelerate launch --num_processes 16`, master
  `jpbo-018-09:30230` (`slurm_logs/bqwen-lm-C0_1694730.out`).
- Suite attempt: `gsm8k,ifeval,humaneval,humaneval_plus,mbpp,mbpp_plus` (per-task
  loop; **only GSM8K started**).
- All 16 ranks built GSM8K contexts (`Building contexts for gsm8k on rank 0…15`).
- `Running generate_until requests` at **2026-09-06 19:53:14**.
- Decode pins: baseline (no DualCache); `max_new_tokens=512`, `num_steps=32`.

### 5.2 Generation largely worked

Evidence:

- **1328** `question:` / `answer:` print pairs in the Slurm out (GSM8K test is
  1319 items — consistent with near-complete sharded generation plus light
  padding).
- Rank 0 wrote  
  `outputs/.../1660579/lm_eval/tok_s_lm_eval.json`:  
  `tokens_generated=34093`, `elapsed_s=16729.8` (**4.65 h**), `tok_s≈2.04`.
- Answers are English CoT-ish (not C2 `新人玩家` loops) — another soft check
  that C0 decode under chat templates is alive.

**Missing:** any `results_*.json`, `SUMMARY.json`, or `exact_match` lines. The
job never reached metric aggregation / file write.

### 5.3 Failure mode (root cause)

At **~00:39–00:41** (≈10 minutes after rank 0 finished generate), multiple ranks
logged:

```text
Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=2, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
```

Then `SIGABRT` / `ChildFailedError` / job **FAILED** exit 1.

lm-eval’s multi-GPU path (`lm_eval/evaluator.py`) does:

1. `accelerator.gather` on per-rank instance counts (pad to equal request counts).
2. `generate_until` **independently** on each rank (no per-example barrier).
3. **`lm.accelerator.wait_for_everyone()`** after responses return.
4. Later `torch.distributed.gather_object` for metrics (never reached here).

With BlockSampler at **~2 tok/s** and up to **512** new tokens, one example can
take **several minutes**. Equal request counts still allow **large wall-clock
skew** across ranks (short vs long CoT / early stop). Finished ranks enter
`wait_for_everyone()` (NCCL allreduce of a scalar); stragglers still inside
`generate_until`. Default NCCL watchdog **600 s** fires → whole job dies.

So this is **not** “C0 cannot do GSM8K” and **not** a bad checkpoint load. It is
**distributed sync timeout << decode-time variance** on a slow diffusion sampler.

Accelerate also printed the confusing default warning (“More than one GPU was
found… pass `--num_processes=1`”) even though we passed `--num_processes 16`;
that warning is a red herring relative to the ALLREDUCE abort.

### 5.4 What `1694730` does / does not confirm

| Claim | Status |
|-------|--------|
| Multi-node Accelerate launch + ckpt load + sharded GSM8K generate | **Mostly yes** (until barrier) |
| End-to-end lm-eval SUMMARY / scores | **No** |
| Fast-dLLM DualCache / fused throughput | **Not tested** |
| Full suite (IFEval, HumanEval±, MBPP±) | **Not reached** |

---

## 6. Interpretation

Pipeline sanity is **split**:

1. **Train path after `plain_ce` sign fix:** healthy (`1694554` + `1698982`).
2. **Eval path:** single-process logic and BlockSampler chat generate look fine
   on C0; **multi-node gather is brittle** at current tok/s without a long
   distributed timeout (or fewer ranks / shorter `max_new_tokens`).

Until lm-eval completes with written metrics, we still lack the C0 chat-eval
roof numbers the thesis needs.

---

## 7. Code / harness changes (this morning)

| Path | Change |
|------|--------|
| `src/discrete_diffusion/evaluations/block_qwen_lm_eval.py` | `Accelerator(InitProcessGroupKwargs(timeout=…))`; default **6 h** via `LM_EVAL_DIST_TIMEOUT_SEC` |
| `scripts/slurm/lm_eval.sbatch` | Export `LM_EVAL_DIST_TIMEOUT_SEC`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, `NCCL_TIMEOUT` (default 21600) |
| `examples/block_qwen/lm_eval.sh` | Same env defaults for non-sbatch launches |

Optional next ops (not all done here): resubmit C0 with the new timeouts;
consider `NUM_NODES=1` smoke; lower `MAX_NEW_TOKENS` for faster sync; silence
per-example stdout on non-zero ranks to cut I/O.

---

## 8. Job genealogy (this incident)

| Job ID | Role | State | Notes |
|--------|------|-------|-------|
| `1694554` | fastdllm_v2 train | FAILED* | *teardown NCCL after successful step 1900 |
| `1698982` | offline eval of 1694554 | COMPLETED | ELBO/gen-PPL/samples OK |
| `1694730` | C0 lm-eval 4×4 | FAILED | GSM8K generate ~done; `wait_for_everyone` 600s NCCL timeout |

Invalidated for score tables: none new beyond prior C2/poisoned runs.  
**Do not discard** `1694554` checkpoints because of Slurm FAILED.

---

## 9. Open questions / next

1. Resubmit C0 lm-eval with raised timeouts (same 4×4 or 1×4 first).
2. After C0 SUMMARY exists, launch fastdllm `1694554` lm-eval with
   `DECODE_PROFILE=dual_cache UNMASK_THRESHOLD=0.9 FORCE_GREEDY=1`.
3. Decide whether per-rank generate progress logging should be rank0-only.
4. Prefetch gaps (GPQA/MMLU/MATH) still block a full paper-shaped average.

---

## 10. Thesis scrapbook

**Distributed evaluation of slow samplers.**  
When each rank runs an independent, long-running ancestral decoder and the
harness only synchronizes after the full local shard finishes, the barrier
timeout must be sized to **worst-case per-rank wall time skew**, not to
training-step collectives. A 10-minute NCCL watchdog is appropriate for dense
AllReduce during optimization and inappropriate for multi-minute-per-example
block diffusion decode. Observing near-complete sample logs without metric
files is the signature of “generate succeeded, gather timed out.”

**Operational reading of Slurm states.**  
A multi-node Lightning job can write `last.ckpt`, enqueue follow-on eval, and
still exit non-zero during NCCL teardown. Checkpoint validity should be judged
from artifacts and loss curves, not from `State=FAILED` alone.

---

## 11. Status board

| Item | State |
|------|--------|
| `plain_ce` train sanity (`1694554`) | **Confirmed** (positive loss, ckpt + offline eval) |
| Slurm FAILED on `1694554` | **Teardown noise** — keep ckpt |
| C0 GSM8K generate | **Mostly worked** (~2 tok/s, English CoT) |
| C0 lm-eval scores | **Missing** — NCCL barrier timeout |
| Timeout harness fix | **Landed** (6 h defaults) |
| Resubmit C0 lm-eval | **Pending** |

### Updates

**Resubmit:** Slurm **`1700915`** (`bqwen-lm-C0`), 4×4, `--time=12:00:00`, same
partial suite `gsm8k,ifeval,humaneval,humaneval_plus,mbpp,mbpp_plus`,
`LM_EVAL_DIST_TIMEOUT_SEC=21600` (+ matching NCCL heartbeat). Logs:
`slurm_logs/bqwen-lm-C0_1700915.{out,err}`. (24h wall rejected by QOS;
12h may finish GSM8K + start later tasks — extend via follow-up jobs if needed.)

### Gap close (same morning) — both eval stacks wired + routed

Closed the earlier honesty gaps:

1. **`generate_samples` now applies decode overrides** into the checkpoint
   config (`sampling.*` / `hydra_overrides`). Previously
   `arpc_decode_eval.sbatch`’s `use_arpc=true` was a no-op.
2. **`arpc_decode_eval.sbatch`** pins BlockGen ARPC explicitly:
   `use_arpc=true arpc_mode=blockgen corruption=divergence prefix_fill=false`.
3. **Family router:** `scripts/submit_family_eval.sh` +
   `scripts/_infer_block_qwen_eval_profile.bash`
   - masked + DualCache/confidence pins → chat lm-eval
     `DECODE_PROFILE=dual_cache UNMASK_THRESHOLD=0.9 FORCE_GREEDY=1`
   - masked without those pins → chat lm-eval `baseline` (C0 floor)
   - uniform/ARPC → `eval_checkpoint` + `arpc_decode_eval` (refuses chat lm-eval)
4. **`lm_eval.sbatch`:** `DECODE_PROFILE=auto` uses the same infer path.
5. **`submit_paper_cell.sh lm_eval|family_eval`** calls the router.

**Submitted exactness:** **`1700927`** (`bqwen-lm-fdllm`) on
`ar2block_masked_1694554` with DualCache / thr=0.9 / greedy (alongside C0
baseline **`1700915`** still running).

### Shared ancestral baseline (policy lock)

`DECODE_PROFILE=baseline` is now the **universal fair core** for all pipelines:

- `BlockSampler`, `steps=32`, **no** ARPC, **no** `unmask_threshold`, **no** DualCache
- Explicitly **clears** paper overlays baked into fastdllm/BlockGen ckpt configs
  (previously “baseline” meant “add nothing”, so fastdllm still ran confidence)

Paper overlays stay recipe-specific: `dual_cache`+thr=0.9 (Fast-dLLM exactness),
`arpc_blockgen` (BlockGen). Optional shared speed: `hierarchical`.

### Free-sampling fix (family-aware)

`generate_samples` / `run_block_qwen_eval` now default to:

- `decode_profile=baseline` (ancestral clears)
- `sample_mode=auto`:
  - **conversion** (Nemotron / ar2block) → `conversion_free`: system +
    `<|im_start|>assistant\\n` prefix (not bare BOS)
  - **native** (OWT / block) → `native_free`: prior + inject_bos LM free-gen
  - `bare_bos` kept as explicit ablation

Writes `samples.meta.json` recording mode/prefix. Text export stops at
`<|im_end|>` when present.

### Adopted as project default (2026-09-07)

Official policy locked in:

- `.cursor/rules/sampling-family-policy.mdc` (`alwaysApply`)
- `configs/eval/generate_samples.yaml` defaults
- `examples/block_qwen/eval.sh`, `tools/run_block_qwen_eval.py`
- `experiments/2026-08-31_postfix/RUN_PROTOCOL.md` §1–3
- ARPC path: `native_free` + `decode_profile=keep` + ARPC pins

`Diffusion-new` is canonical; do not expect the sibling `Diffusion/` tree to
mirror this unless explicitly synced.

---

## Harsh audit — entire eval / sampling pipeline (2026-09-07)

**Verdict:** Policy code exists, but **offline headlines are still pre-policy
artifacts**. Adopted `sample_mode=auto` / `decode_profile=baseline` barely
runs on dirs you already care about because `samples.pt` is reused forever.

### Critical

1. **Stale bare-BOS reuse** — `eval.sh` / `run_block_qwen_eval.py` skip regen
   if `samples.pt` exists; no meta check. Live `1660579` / `1694554` eval
   tensors are still bare BOS `(64,2048)`, **no `samples.meta.json`**.
2. **No in-loop EOS stop; `max_new_tokens=null` → fill 2048** — `stop_at_im_end`
   only trims `.txt`. Collapse then soups for thousands of tokens.
3. **`first_chunk_only` gen-PPL launders collapse** — e.g. fastdllm PPL 72 on
   1664 tokens after early `im_end`; looks “good”, samples are trash.
4. **Train / SampleSaver still `model.generate_samples()` bare BOS** —
   `algorithms/base.py`; val collapse gate ≠ offline `conversion_free`.

### High

5. **ARPC sbatch hardcodes `native_free`** even for Instruct/transfer data.
6. **`decode_profile=baseline` last-wins defeat** if nested `sampling.*`
   re-enables thr/ARPC after clears.
7. **DualCache/thr baked into fastdllm ckpt** — easy to mix ancestral vs
   exactness samples in one `eval/` folder without meta.
8. **Family inference fragile** — `block_*` run dirs can still be Nemotron;
   auto saved by data name, not folder name.

### Medium / Low

9. Multi-node lm-eval: timeout raised, but 12h may not finish suite; shared
   log tee races; TASKS comma/`--export` footgun.
10. No unit tests for `sample_mode` / profile clears / meta gate.
11. Docs vs artifacts drift (`PAPER_EXPERIMENTS` claims family-aware; disks don’t).

### What still produces bad samples *with* the adopted policy

Reuse of old `samples.pt` · full-length free-gen without EOS break · train
val bare BOS · ARPC `native_free` on Instruct · flattering `first_chunk_only`
PPL · claiming baseline while passing thr via nested overrides.

### Fix order (do next)

1. Hard-invalidate: refuse reuse unless `samples.meta.json` matches mode+profile.
2. Default `max_new_tokens` 256–512; early-stop block loop on EOS/`im_end`.
3. Honest gen-PPL: emit eos_rate/tokens; refuse headline without meta.
4. One API: `base.generate_samples` + SampleSaver honor `sample_mode`.
5. Tests + single `_DECODE_PROFILES` source of truth.

**Until 1–3:** do not trust numbers under `outputs/block_qwen/*/eval/` without
fresh meta.

---

## Updates — audit fixes landed (2026-09-07, later)

All five audit items above were implemented in-tree. Offline headlines under
existing `outputs/block_qwen/*/eval/` remain **scientifically stale** until a
fresh regen writes `samples.meta.json` (none of the live `samples.pt` trees
had meta at fix time — meta gate will refuse reuse automatically).

### What changed

| # | Fix | Where |
|---|-----|--------|
| 1 | Meta reuse gate | `decode_profiles.should_reuse_samples`; wired in `examples/block_qwen/eval.sh` (`FORCE_REGEN`) and `tools/run_block_qwen_eval.py` (`--force-regen`) |
| 2 | EOS early-stop + length cap | `BlockSampler.stop_on_eos`; default `max_new_tokens=512` in `configs/eval/generate_samples.yaml` + `configs/sampling/block.yaml` |
| 3 | Honest gen-PPL | `generative_ppl` emits `eos_rate` / `mean_tokens_before_eos` / `honesty_warning`; `gen_ppl_block_qwen.yaml` sets `require_samples_meta=true` (fail closed) |
| 4 | Unified generate API | `algorithms/base.py::generate_samples(sample_mode=..., max_new_tokens=...)`; `SampleSaver` + `_run_generate_samples_overrides.py` honor it |
| 5 | Shared profiles + tests | `evaluations/decode_profiles.py` is SoT; throughput/lm-eval/generate_samples import it; ARPC sbatch uses `sample_mode=auto`; `tests/test_decode_profiles.py` |

Also: `decode_profile=baseline` now **wins over nested `sampling.*`** in
`generate_samples._collect_sampling_overrides` (profile after nested, before
explicit `hydra_overrides`), so thr/ARPC/DualCache cannot sneak back through
nested Hydra while claiming baseline.

### Tests

```
PYTHONPATH=src python -m pytest tests/test_decode_profiles.py \
  tests/test_block_sampler.py tests/test_gen_ppl_eos.py -q
# 17 passed
```

### Operator notes

- Regen C0 / fastdllm free-gen: `FORCE_REGEN=1 bash examples/block_qwen/eval.sh <ckpt>`
  (or delete `eval/samples.pt` — missing meta alone also forces regen).
- Do **not** trust pre-fix `gen_ppl_metrics.json` beside bare-BOS tensors.
- lm-eval jobs still running at fix time: **1700915** (C0 baseline),
  **1700927** (fastdllm DualCache exactness), plus **1701099** (`lever_C2`).
  Those harness paths are independent of offline `samples.pt` reuse; still
  paste SUMMARYs into Updates when they finish.

### Updates — lm-eval cancel + resubmit (2026-09-07 ~08:24)

Cancelled mid-run **`1700915`** / **`1700927`** (fdllm had already SIGBUS’d
at ~08:00 on rank 14). Left train **`1701099`** alone.

Resubmitted same recipes via `submit_family_eval.sh --lm-eval-only`:

| New job | Name | Ckpt | Profile |
|---------|------|------|---------|
| **1701209** | `bqwen-lm-C0` | `ar2block_masked_1660579` | baseline |
| **1701210** | `bqwen-lm-fdllm` | `ar2block_masked_1694554` | dual_cache + thr=0.9 + greedy |

Tasks unchanged: `gsm8k,ifeval,humaneval,humaneval_plus,mbpp,mbpp_plus`,
4×4, 12h, dist timeout 21600s. Logs:
`slurm_logs/bqwen-lm-C0_1701209.{out,err}`,
`slurm_logs/bqwen-lm-fdllm_1701210.{out,err}`.

### Updates — C2 resume (2026-09-08)

Train **`1701099`** hit wall TIMEOUT at ~step **5501** / Epoch 2 (~19%), loss
~1.0–1.3; `last.ckpt` + `2-5500.ckpt` present under
`outputs/block_qwen/ar2block_masked_1701099/`.

Resumed into the **same** run dir via `./scripts/resume_block_qwen.sh`:

| Field | Value |
|-------|--------|
| Job | **1718374** (`resume_ar2block_masked`) |
| From | `.../1701099/checkpoints/last.ckpt` (global_step **5501**) |
| To | `trainer.max_steps=6000` (**499** steps remain) |
| Recipe | C2_fdllm: shift + complementary fused + `mask_schedule=fast_dllm` + `plain_ce` |
| Queue | PD — `ReqNodeNotAvail, Reserved for maintenance` |

### Updates — lm-eval code-suite resume (2026-09-08)

**1701209 / 1701210** already wrote GSM8K + IFEval, then died on multi-rank
`code_eval` arrow races / wall clock. Resubmitted **remaining code tasks only**
(keep prior scores on disk):

| Job | Name | Ckpt | Profile | Nodes |
|-----|------|------|---------|-------|
| **1718375** | `bqwen-lm-C0-code` | `1660579` | baseline | 1×4 (less code_eval race) |
| **1718376** | `bqwen-lm-fdllm-code` | `1694554` | dual_cache + thr=0.9 | 1×4 |

`TASKS=humaneval,humaneval_plus,mbpp,mbpp_plus`, `SKIP_THROUGHPUT=1`, 12h.
Also PD behind maintenance with the C2 resume.

### Updates — default width 8×4 (2026-09-08)

Paper/train + family lm-eval defaults raised **4→8 nodes** (`submit_lever`,
`submit_family_eval`, `lm_eval.sbatch`, `submit_fastdllm`,
`submit_blockgen_owt`). Micros / smoke / single-node decode jobs unchanged.

Cancelled PD **1718374/75/76** and resubmitted at 8×4 (C2 resume needs
`ALLOW_RESOURCE_MIGRATION=1` because original run was 4×4; GBS 256 fixed):

| Job | Role | Nodes |
|-----|------|-------|
| **1718377** | C2 resume 5501→6000 | 8×4 |
| **1718378** | C0 code lm-eval | 8×4 |
| **1718379** | fastdllm code lm-eval | 8×4 |

### Updates — policy free-gen regen (2026-09-08)

Submitted 1-GPU upstream eval (`--force-regen`, `sample_mode=auto`,
`decode_profile=baseline`, `max_new_tokens=512`) into fresh dirs (old bare-BOS
`eval/` kept for contrast):

| Job | Ckpt | Out |
|-----|------|-----|
| **1718384** | C0 `1660579` | `.../1660579/eval_policy/` |
| **1718385** | fastdllm `1694554` | `.../1694554/eval_policy/` |

Judge sampling from `samples.txt` + `samples.meta.json` + honest gen-PPL there
once jobs finish (also PD behind maintenance).

### Updates — login smoke (2026-09-08)

2×256-token `conversion_free`+`baseline` smokes on login GH200 →
`eval_policy_smoke/` for C0 + fastdllm. Meta correct; `.txt` misleading
(`stop_at_im_end` cuts at prefix `<|im_end|>`). Raw decode of the denoised
span (pos 24…~280): **still high-entropy soup** on both ckpts (no EOS, not
assistant-fluent). **Open free-gen not solved** by the header alone; wait on
full Slurm regen only for stats, not for a different qualitative verdict.
