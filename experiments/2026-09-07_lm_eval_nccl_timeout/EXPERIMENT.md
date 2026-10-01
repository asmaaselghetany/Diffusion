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

### Updates — deep dive: what was actually wrong (2026-09-08)

**Live probes (login GH200)** on C0 `1660579` and fastdllm `1694554`:

1. **Open `conversion_free` (no user)** — first-token flat / OOD (`</`,
   `think`, …); 256-tok ancestral span = soup. **Not a fair conversion meter.**
2. **User prompt `What is 2+2?` + greedy** — fastdllm **answers correctly**
   (`**4**` / `! 4.` under DualCache+thr=0.9). C0 weaker/loopy but not
   `新人玩家` collapse. So **conditional chat decode works**; open-header
   free-gen was the false crisis.
3. First-pos logits: fastdllm ranks **`4`** under the math user prefix;
   open header does not.

**Ruled out as primary:** prefix not frozen; `max_new` broken; train sign
bug on `1694554`; DualCache required for *any* correct short math (baseline
greedy already works on fastdllm).

**Still real:** ancestral open free-gen looks bad; C0 chat math weak vs
fastdllm; GSM8K 18% was DualCache exactness — don’t claim baseline recipe
alone without a baseline lm-eval; `.txt` truncate-at-prefix-`im_end` bug.

**Paper path under time pressure:** roof = **chat-conditional** lm-eval
(+ optional labeled DualCache). Demote open-header free-gen to ablation.
Do not burn more wall on “fix free-gen soup” as if the model can’t generate.

### Updates — near-full Fast-dLLM suite queued (2026-09-08)

Cancelled low-value **1718384/85** (open free-gen) and partial code **1718378/79**.

`submit_family_eval.sh` default suite is now **`fastdllm`** (empty `TASKS` →
full list from `lm_eval.sh`).

Prefetch: completed **MMLU** (`hails/mmlu_no_train` all 57 subjects). Still
blocked offline:
- **GPQA** — Hub gated (needs authenticated `HF_TOKEN`)
- **minerva_math** — `math_verify` requires Python ≥3.10 (venv is 3.9)

Submitted near-full conversion roofs (8×4, 12h, skip throughput):

| Job | Ckpt | Profile | Tasks |
|-----|------|---------|-------|
| **1718425** | C0 `1660579` | baseline | `mmlu,gsm8k,ifeval,humaneval(+),mbpp(+)` |
| **1718426** | fastdllm `1694554` | DualCache+thr0.9+greedy | same |
| **1718377** | C2 resume | train | still PD |

Label tables as Fast-dLLM suite **minus GPQA/Minerva** until those deps land.

### Updates — close closable Fast-dLLM gaps (2026-09-09)

Closed every **operational** gap we control. Irreducible remain: Nemotron
subset undisclosed, fused CUDA ≠ our DualCache port, seeds/hardware.

**Defaults wired**
- `MAX_NEW_TOKENS` default **2048** (`lm_eval.sh` / `lm_eval.sbatch` /
  `submit_family_eval`) — Hub `eval.py` parity (was 512).
- Fastdllm family exactness default thr **1** (paper §4 accuracy; parallel
  decode off). Hub/speed thr=0.9 via `FORCE_UNMASK_THRESHOLD` or
  `PAPER_BOTH_THR=1` (separate `OUT_DIR`).
- `DECODE_PROFILE=dual_cache` pins **`sub_block_size=8`** (Hub
  `small_block_size`).
- `submit_fastdllm.sh` default **`MAX_STEPS=6000`** (~3.15B; appendix 1.5B
  schedule). Short probe: `MAX_STEPS=1900`.
- Auto `OUT_DIR=…/lm_eval_m{max}_{profile}[_t{thr}]` so cells don’t clobber
  old 512 results.

**Jobs**
- Cancelled stale held **1718425/26** (`max_new=512`).
- Released C2 resume **1718377** (5501→6000 paper schedule).
- Resubmitted Hub-parity evals (8×4, suite minus GPQA/Minerva):

| Job | Ckpt | Decode | OUT_DIR tag |
|-----|------|--------|-------------|
| **1725187** | C0 `1660579` | baseline | `lm_eval_m2048_baseline` |
| **1725188** | fastdllm `1694554` | DualCache+**thr=1**+greedy | `…_t1` |
| **1725189** | fastdllm `1694554` | DualCache+thr=0.9+greedy | `…_t0.9` |
| **1718377** | C2 `1701099` | train resume | — |

**Still open (needs deps, not script knobs):** GPQA HF token; Minerva
Python ≥3.10; finished C2@6000 then re-eval that ckpt under the same
`m2048` protocol.

### Updates — Hub 1.5B paper-table calibration (2026-09-09)

**Goal:** run official `Efficient-Large-Model/Fast_dLLM_v2_1.5B` through
**their** `v2/eval.py` (not UNI-D2 BlockSampler) to bound meter vs train gap.

**Why separate stack:** Hub modeling needs `transformers≥4.53.1`; train
venv locked at **4.45**. Dedicated venv:
`/e/scratch/scifi/elsayed3/.venvs/hub_fastdllm` (torch 2.6.0+cu126).
Weights snapshot:
`/e/scratch/scifi/elsayed3/hf_cache/models/Fast_dLLM_v2_1.5B`.

**Wiring:** `scripts/submit_hub_fastdllm_eval.sh`,
`scripts/slurm/hub_fastdllm_lm_eval.sbatch`,
`examples/block_qwen/hub_fastdllm_lm_eval.sh`.

**Submitted (4×4, tasks `mmlu,gsm8k,ifeval`, max_new=2048):**

| Job | Role | OUT_DIR |
|-----|------|---------|
| **1725445** | Hub 1.5B thr=**1** (paper accuracy) | `outputs/hub_fastdllm_eval/Fast_dLLM_v2_1.5B_t1` |
| **1725446** | Hub 1.5B thr=0.9 | `…_t0.9` |

**Readout when done:** if Hub ≈ paper (GSM8K~62 / IFEval~47 / MMLU~55) →
our recreate gap is train/data; if Hub ≪ paper → eval/env still broken
even on released weights.

**Note:** earlier paper-parity recreate jobs **1718377 / 1725187–89**
FAILED on Jupiter `PSI: doSpawn` (cluster spawn), not recipe — resubmit
separately when nodes are healthy.

### Updates — full day ops + science gate (2026-09-09 afternoon)

#### Cluster / launch hardening
- Flaky first `srun` (`PSI: doSpawn` / Hangup): `scripts/_srun_retry.bash` +
  warmup hostname barrier; wired into train, recreate lm-eval, Hub sbatch.
- `srun_retry` only retries signal failures **within 120s** — mid-suite
  Force Terminated must **not** redo the whole paper eval.
- Train DDP: restore **shared GPUs** —
  `trainer.devices=GPUS_PER_NODE` + `CUDA_VISIBLE_DEVICES=0,1,2,3`
  (`scripts/_block_qwen_ddp.bash`). Exclusive `devices=1` broke world-size.
- Family default suite → **`paper_acc`** = `mmlu,gsm8k,ifeval`
  (`submit_family_eval.sh`). Drop humaneval from the core paper path
  (code_eval races / wipe risk).

#### C2 paper-schedule train — DONE
| Job | Role | Result |
|-----|------|--------|
| **1726863** | resume `ar2block_masked_1701099` 5501→6000 | **COMPLETED** (~35m) |

Ckpts: `…/ar2block_masked_1701099/checkpoints/{2-6000,last}.ckpt` (~11:42).
Recipe = `C2_fdllm_full` (shift + complementary fused + `mask_schedule=fast_dllm`
+ `plain_ce`). **Do not lm-eval C2@6000 until Hub/C0 gate below.**

Prior failed/cancelled resumes (PSI / DDP / cancel): 1718377, 1725764,
1726191, 1726486, 1726665, 1726727 — superseded by 1726863.

#### Skeleton clarification (C0 vs fastdllm)
Same conversion skeleton: `LINE=ar2block`, arm `masked`, Qwen→block, Nemotron.
| Preset | Run | Levers |
|--------|-----|--------|
| **C0** | `1660579` | `levers: []` — ELBO / `alpha`; eval **baseline** |
| **fastdllm_v2** | `1694554` (~1B / 1900) | shift, complementary, `fast_dllm`, `plain_ce` (+ decode DualCache) |
| **C2_fdllm_full** | `1701099` @6000 | same train levers as fastdllm, paper step budget |

#### Recreate fastdllm paper lm-eval (cancelled after scores + collapse)
Jobs **1726193** (thr=1) / **1726194** (thr=0.9) ran ~1h13 then FAILED
(exit 1) while retrying after humaneval; **cancelled** to stop wipe.
Scores saved under `…/1694554/lm_eval_m2048_dual_cache_t{1,0.9}/`:

| | thr=1 | thr=0.9 |
|--|-------|---------|
| MMLU | **48.3%** | 48.3% |
| GSM8K strict / flex | **0% / 0%** | **0% / 0%** |
| IFEval prompt-strict / inst-loose | ~15.9% / ~32.0% | ~16.8% / ~32.7% |

**GSM8K=0 root cause:** generative collapse — every sample is essentially
`**The answer is (A)**!\boxed{A}` (not an extract-metric bug). MMLU
loglikelihood still “works.”

#### Hub 1.5B calibration — failure ladder + partial paper scores
Many Hub submits died on ops before science. Ladder (fix → next fail):

1. PSI first-srun → `srun_retry` + warmup.
2. Missing `cais/mmlu` subject configs → prefetch **all 59**; submit gate.
3. `jinja2` missing/old → hub venv **jinja2 3.1.6**.
4. `gather_object` / torch 2.6: `gather_list` on non-dst ranks —
   patched lm_eval `evaluator.py` (`object_gather_list=… if RANK==0 else None`)
   **still failed** because Hub `eval.py` used
   `accelerator.local_process_index` (0 on every node’s GPU0).
5. **Fix:** `self._rank = self.accelerator.process_index` in
   `third_party/Fast-dLLM/v2/eval.py`; gather buffers also keyed off
   `torch.distributed.get_rank()`.
6. After MMLU+GSM8K success: IFEval `ModuleNotFoundError: langdetect` →
   `pip install langdetect immutabledict nltk` + submit preflight in
   `submit_hub_fastdllm_eval.sh`.

**Partial Hub results** (jobs **1727643/44**; IFEval crashed; scores on disk
under `outputs/hub_fastdllm_eval/Fast_dLLM_v2_1.5B_t{1,0.9}/`):

| | thr=1 | thr=0.9 | Paper (1.5B ref) |
|--|-------|---------|------------------|
| MMLU | **53.5%** | 53.5% | ~55% |
| GSM8K flexible | **62.8%** | **62.9%** | ~62% |
| GSM8K strict | 7.1% | 6.8% | (paper reports flex-style) |
| IFEval | pending | pending | ~47% |

**Gate verdict:** official Hub **generates** (GSM8K flex ≈ paper). Recreate
GSM8K=0 is **train/decode/data**, not “our lm-eval meter is dead.”

IFEval-only resubmits (deps fixed): **1727734** (thr=1), **1727735** (thr=0.9)
— running as of ~12:27.

#### C0 paper lm-eval
| Job | State | Notes |
|-----|-------|-------|
| **1726192** | **RUNNING** (~1.7h+) | `paper_acc`, baseline, 8×4; MMLU wrote **~49.5%**; still in GSM8K generate |

#### Live jobs (snapshot ~12:27)
| Job | Name | ST |
|-----|------|----|
| 1726192 | bqwen-lm-C0-paper | R |
| 1727734 | hub-fdllm-t1 (ifeval) | R |
| 1727735 | hub-fdllm-t0.9 (ifeval) | R |

#### Next (ordered)
1. Wait Hub IFEval → full paper-table row for released 1.5B.
2. Finish C0 **1726192** SUMMARY (esp. GSM8K gens vs fastdllm collapse).
3. Only then eval **C2@6000** under same `m2048` DualCache protocol.
4. Still open deps: GPQA HF token; Minerva needs Py≥3.10; Nemotron
   internal subset unmatched (authors: no reasoning data).

### Updates — GSM8K≈0 was prompt wipe, not model (2026-09-09 evening)

**Root cause:** `block_qwen_lm_eval._generate_batch` used
`max_prefix = seq_len - max_new`. Conversion ckpts have **`length=2048`**.
With Hub-parity `MAX_NEW_TOKENS=2048` that set **`max_prefix=1`**, so every
GSM8K prompt was reduced to a single token → near-identical garbage
continuations (C0 flex **2%**, fastdllm m2048 flex **0%** / `\boxed{A}`).

**Evidence**
- Older C0 runs at `max_new=512` (`1694730` etc.): diverse English CoT.
- This run `1726192` `max_new=2048`: ~43 unique answer prefixes / 1300+
  dumps; batches of 32 identical strings.
- Hub 1.5B is fine at 2048 because `max_position_embeddings=32768`.

**Fix:** prefer keeping the full prompt; **clamp `max_new`** to
`seq_len - longest_prefix` instead of destroying the question
(`block_qwen_lm_eval.py`). Comment in `lm_eval.sh`.

**Invalidated:** C0 `lm_eval_m2048_baseline` GSM8K (and in-flight IFEval);
fastdllm `lm_eval_m2048_dual_cache_t{1,0.9}` generative scores. MMLU
loglikelihood unaffected. Cancelled **1726192**.

**Resubmit** paper_acc (C0 + fastdllm DualCache) after fix — clamp will
allow ~prompt_len + gen ≤ 2048 (effective max_new ≈ 1800+ for short
GSM8K chats). Optional: `MAX_NEW_TOKENS=512` for apples-to-apples with
pre-m2048 C0 roofs.

**Resubmitted (prompt-wipe fix, fresh `*_fix` OUT_DIRs):**

| Job | Role | OUT_DIR tag |
|-----|------|-------------|
| **1733454** | C0 baseline paper_acc | `lm_eval_m2048_baseline_fix` |
| **1733457** | fastdllm DualCache thr=1 | `…_dual_cache_t1_fix` |
| **1733459** | fastdllm DualCache thr=0.9 | `…_dual_cache_t0.9_fix` |

Full-suite redos **cancelled** (MMLU already valid). Generative-only
`TASKS=gsm8k,ifeval` resubmit:

| Job | Role |
|-----|------|
| **1733620** | C0 baseline gen |
| **1733621** | fastdllm DualCache thr=1 gen |
| **1733622** | fastdllm DualCache thr=0.9 gen |

**C2@6000 gen probe** (`1701099/last.ckpt`, same wipe-fixed DualCache
protocol, `TASKS=gsm8k,ifeval`):

| Job | Role |
|-----|------|
| **1733979** | C2 DualCache thr=1 |
| **1733980** | C2 DualCache thr=0.9 |

**C2@6000 RESULTS** (jobs completed; wipe-fixed DualCache):

| | GSM8K flex | IFEval prompt-strict |
|--|------------|----------------------|
| fastdllm `1694554` (~1B) thr=1 | 18.8% | 15.9% |
| fastdllm thr=0.9 | 18.9% | 16.8% |
| **C2@6000** thr=1 | **20.3%** | **18.9%** |
| **C2@6000** thr=0.9 | **23.0%** | **18.9%** |
| Hub 1.5B | ~63% | ~45% |

Step budget 1900→6000: **~+2–4 pp GSM8K**, **~+2–3 pp IFEval** — not Hub territory.
C0 gen **1733620** still running (lever floor).

### Updates — why scores stay bad (2026-09-09 evening)

Hub gate still valid (~63 / ~45). C2@6000 + latest train recipe still
~20–23% GSM8K. Investigation:

**Not the meter / not missing train levers / not step count alone.**

| Evidence | Detail |
|----------|--------|
| Sample pathology | C2/fastdllm gens: heavy **`!` (token id 0)** mid-words (`20!000`, `!think>` = corrupted `</think>`); ~5% `\boxed{}`. Old C0@512: **0 bangs**, clean CoT. |
| Hub gens | Long CoT, ~1300 `\boxed{}` in gsm8k log chunk; almost no think-soup. |
| Mask id | Both use **151665** (`\|<MASK>\|` / next-id alloc) — not a mask-id mismatch. |
| Hub decode | Paper calib runs with **`use_block_cache=False`** + confidence thr. Our C2 roof used **DualCache ON**. |
| Levers effect | C0→fastdllm ~+17 pp GSM8K only; IFEval flat ~17%. |

**Hypothesis:** residual gap is **weights/data + generate-stack** (BlockSampler port vs Hub `batch_sample`), visible as token-0 corruption under our confidence/DualCache path — not “forgot a train lever.”

**Probe queued:** C2@6000 with **Hub-matched decode** (baseline profile = DualCache off + `FORCE_UNMASK_THRESHOLD=1` greedy): job **`bqwen-lm-C2-hubmatch`**. If scores jump, DualCache port is implicated; if still ~20%, weights/data dominate → next = Hub weights on our sampler.

### Updates — DualCache × shift → token-0 (2026-09-09 deep dive)

**Root mechanism for `!` spam (id 0):**

1. DualCache `replace` returns logits only in `[w0:w1)` and **zeros elsewhere**.
2. `align_shift_logits` did a **full-sequence** cat shift → `aligned[w0] = logits[w0-1] = 0-vector`.
3. Mask/pad banned → greedy `argmax` on zeros → **token 0 (`!`)** at every window start.
4. Confidence **force-max over the full sequence** could also “commit” future MASK positions that are then discarded when writing back `[start:end]` → **hubmatch** (`1734232`) crashed with `masks N→N` (no progress).

Hub shifts the **window-sized** logit tensor (pos0 keeps its own column). Train block-causal mask matches Hub; mask id **151665** matches; embed head is **151666** vs Hub **151936** (secondary).

**Fix landed** (`shift_logits.py` + `block_sampler.py`): window-local shift + confidence commit scoped to active window. Unit-checked.

**Resubmit after fix:**

| Job | Decode |
|-----|--------|
| **1734404** `bqwen-lm-C2-dc-fix` | DualCache thr=1 (bang path) |
| **1734406** `bqwen-lm-C2-hubmatch2` | baseline + thr=1 (Hub-like) |

Expect: far fewer mid-word `!`; scores may move if decode was the bottleneck. Vocab shrink / data / missing AR block-bridge still open if scores stay ≪ Hub.

### Updates — full decode audit + fix pack (2026-09-09 evening)

**Symptom after first shift “fix”:** DualCache GSM8K **20.3% → 2.0%**; bangs gone but gens open with Nemotron tag debris (`think>` / `thinkthink>`). Same debris on hubmatch2 baseline — over-correction, not DualCache-only.

**Root cause:** window-local shift was applied on **every** denoise window (including dense/prefill). Train/Hub need full-span shift then slice; window-local is **only** correct on DualCache replace (zero-padded outsides).

**Fix pack landed** (unit-tested):

| Fix | Where |
|-----|--------|
| `align_shift_logits(mode='full'\|'window')` — window only on DualCache replace | `shift_logits.py`, `block_sampler._logits` |
| `active_end` = attention-block ceil under hierarchical / DualCache / single-stream / confidence | `_truncated_active_end` + `_use_block_scope` |
| DualCache lifetime: keep across sub-windows; Hub refresh on MASK@window_start / `active_len` mismatch; clear per attn block | `_maybe_refresh_dual_cache`, `generate` |
| Hub AR block-bridge (unshifted `argmax` at `end-1`); `_init_block` preserves seed | `_ar_block_bridge` |
| Mid-block EOS stop: EOS present and **no MASK before it** | `_eos_stop_ready` |
| `ban_mask_pad_logits` (default on; set false for Hub bit-parity probes) | `configs/sampling/block.yaml` |
| New decode profile **`hubmatch`**: hierarchical + single_stream + sub8 + DualCache **off** + greedy | `decode_profiles.py` |

**Still not bit-exact Hub:** past-KV shape (full prefix vs Hub past+block), replace attn-mask shortcut; residual data/slice deltas after enhanced baseline.

**Re-eval after this pack:** use `DECODE_PROFILE=hubmatch` (not bare `baseline`) + thr, and DualCache with the new shift modes. Expect GSM8K back near ~20% (not ~2%) if decode was the crash; Hub ~63% still needs data/weights work.

**Queued (2026-09-09 ~22:01):** C2 `1701099` `gsm8k,ifeval` thr=1 greedy=1 8×4:

| Job | Profile | OUT |
|-----|---------|-----|
| **1735005** `bqwen-lm-C2-hubmatch-dp` | `hubmatch` | `…/lm_eval_m2048_hubmatch_t1_decodepack` |
| **1735006** `bqwen-lm-C2-dc-dp` | `dual_cache` | `…/lm_eval_m2048_dual_cache_t1_decodepack` |

Still running (pre-pack / unrelated): **1734406** hubmatch2 (baseline+wrong shift), **1733620** C0-gen.

### Updates — enhanced conversion baseline (not levers) (2026-09-09)

Shared masked+uniform hygiene in
`src/discrete_diffusion/data/conversion_baseline.py` (not `registry.yaml`):

| Change | Effect | Needs retrain? |
|--------|--------|----------------|
| **Train = eval ChatML** (short `You are a helpful assistant.`; `add_generation_prompt`) | lm-eval / free-gen no longer use Alibaba stock system | **Eval-only helps existing ckpts** (they trained on short system) |
| **Hub vocab keep** — never shrink 151936→151666; `_effective_vocab_size=151936` | Softmax/embed geometry matches Hub | **Yes** (new init) |
| **Nemotron defaults** = `chat,safety,science,math,code` (math/code cap 100k) | Math-heavy GSM8K signal on both arms | **Yes** (cache fingerprint `v2-baseline`) |
| Ancestral decode hygiene already in `block.yaml` / `decode_profile=baseline` | Shared floor | No |

Fast-dLLM / ARPC remain overlays. Tests: `tests/test_conversion_baseline.py`.

### Updates — step-0 unblock + resubmit (2026-09-09 ~22:50)

Prior decodepack jobs **FAILED**: `hubmatch` missing from `lm_eval.sh` allowlist; DualCache
hit embed **151666 ckpt vs 151936 model** after Hub-vocab keep.

**Fixes:** allow `hubmatch` in `examples/block_qwen/lm_eval.sh`; peek ckpt embed rows and
force `_effective_vocab_size` on load (`checkpoint_utils` + `conversion_baseline`);
train resume/finetune peeks the same way.

**Resubmitted** C2 `1701099` `gsm8k,ifeval` thr=1 greedy 8×4 (chat template already fixed):

| Job | Profile | OUT |
|-----|---------|-----|
| **1735837** `bqwen-lm-C2-hubmatch-dp` | `hubmatch` | `…/hubmatch_t1_decodepack` |
| **1735838** `bqwen-lm-C2-dc-dp` | `dual_cache` | `…/dual_cache_t1_decodepack` |
| **1735839** `bqwen-lm-C2-base-chatfix` | `baseline` | `…/baseline_t1_chatfix` (vs old hubmatch2 ~10%) |

Compare to pre-pack DualCache ~20% / broken shiftfix ~2% / hubmatch2 ~10%.

### Updates — enhanced-baseline retrain for weights/data (2026-09-09 ~23:05)

Decode fixes plateau ~20% on old weights (no math, vocab shrink). Submitted **paper-scale
C2_fdllm_full** retrain with enhanced conversion baseline:

| | |
|--|--|
| Job | **1735919** `lever_C2_fdllm_full_masked` |
| Levers | shift + complementary fused + `fast_dllm` + `plain_ce` (same as `1701099`) |
| Data | `NEMOTRON_SFT_SPLITS=chat,safety,science,math,code` (math/code ≤100k) |
| Preprocess | `qwen-chat-block-aligned-v2-baseline` (cache rebuild) |
| Vocab | Hub-keep **151936** (new init; not shrink) |
| Chat | shared short-system ChatML train↔eval |
| Scale | 6000 × 256 × 2048, 8×4 |

Audit: `outputs/block_qwen/lever_overrides/C2_fdllm_full_ar2block_masked_job1735919.{txt,data_env.txt}`

After train: lm-eval with `hubmatch` + DualCache. Expect GSM8K move **well above ~20%** if data was the main Hub gap; if still flat → cross-stack Hub generate next.

### Updates — lm-eval on enhanced C2 `1735919` (2026-09-10 ~09:04)

Old C2 decodepack roof: DualCache/hubmatch **GSM8K 39.1%**. Retrain completed (math+code, vocab 151936). Queued:

| Job | Profile | OUT |
|-----|---------|-----|
| **1738889** `bqwen-lm-C2eb-hubmatch` | `hubmatch` thr=1 | `…/ar2block_masked_1735919/lm_eval_m2048_hubmatch_t1` |
| **1738890** `bqwen-lm-C2eb-dc` | `dual_cache` thr=1 | `…/ar2block_masked_1735919/lm_eval_m2048_dual_cache_t1` |

### Updates — enhanced-baseline C0 retrain (2026-09-10 ~09:26)

Fair floor vs C2eb: **no** Fast-dLLM train levers, same data/vocab/chat hygiene.

| | |
|--|--|
| Job | **1739031** `lever_C0_masked` |
| Levers | `[]` (ELBO / α-schedule defaults) |
| Data | chat+safety+science+math+code (caps 100k) |
| Vocab / chat | Hub-keep 151936; short-system ChatML |
| Scale | 6000 × 256 × 2048, 8×4 |

After done: lm-eval `hubmatch` + DualCache vs C2eb `1735919` (decodepack old C2 was GSM8K 39%).

### Updates — C2eb `1735919` lm-eval RESULTS (2026-09-10)

| Profile | GSM8K flex | IFEval strict |
|---------|------------|---------------|
| DualCache thr=1 | **40.1%** | 20.1% |
| hubmatch thr=1 | **38.9%** | 20.9% |

Math@100k + Hub vocab only moved ~+1 pp vs old C2 decodepack ~39%. Hub still ~63%.

### Updates — all-5 skeleton enhancements (2026-09-10 ~09:55)

Implemented the five conversion-skeleton items (not Fast-dLLM levers):

| # | Item | What landed |
|---|------|-------------|
| **1** | Richer math/code | Caps **math 1M / code 500k** (was 100k). Full splits are ~22M/~10M — uncapped rejected. Preprocess **`v3-richmath`**. |
| **2** | Longer gen context | lm-eval extends buffer to **`EVAL_MAX_SEQ_LEN=8192`** (block-aligned) so Hub `max_new=2048` is not clamped for typical prefixes. Train length stays 2048. |
| **3** | C0 decode A/B | Helper `scripts/submit_decode_ab.sh` (hubmatch + DualCache). Run after C0 v3 ckpt exists. |
| **4** | Packing audit | `tools/packing_audit_fastdllm.py` — Hub packing **undisclosed**; our rule = pad-to-diffusion-block with MASK then pack. |
| **5** | Shared stable cache | Fingerprint uses **resolved (split,cap)** not raw env; `submit_lever` writes `.data_env.txt` lock; C0+C2 share `v3-richmath`. |

**Jobs:** cancelled C0 **1739031** (v2/100k). Resubmitted same skeleton:

| Job | Preset | Notes |
|-----|--------|-------|
| **1739851** | C0 | levers `[]`, v3-richmath |
| **1739853** | C2_fdllm_full | shift+comp+fdllm+plain_ce, v3-richmath |

Audit: `outputs/block_qwen/lever_overrides/{C0,C2_fdllm_full}_*_job173985{1,3}.data_env.txt`

After both finish: `./scripts/submit_decode_ab.sh <ckpt>` on each; compare C0 vs C2 under identical data/decode.

### Updates — cross-stack probe launched (2026-09-10 ~14:05)

Goal: stay skeleton-neutral; isolate **weights vs Hub generate**.

| Step | Status |
|------|--------|
| Export our C2eb `1735919` → Hub HF folder (EMA, 338 tensors, V=151936) | **Done** → `outputs/hub_fastdllm_eval/our_ar2block_masked_1735919/` |
| Hub `eval.py` / `batch_sample` on that export (`use_block_cache=False`, thr=1, gsm8k+ifeval) | **Job 1744042** `hub-fdllm-t1` (8×4; replaced 1743849 @4×4) → `outputs/hub_fastdllm_eval/our_ar2block_masked_1735919_t1/` |
| Tools | `tools/export_block_ckpt_to_fastdllm_hf.py`, `scripts/submit_cross_stack_hub_generate.sh` |

**Readout:**
- If Hub-generate on our weights → ~60%: our BlockSampler is the gap (fix decode; keep train skeleton).
- If still ~40%: gap is data/weights (not generate). Stay baseline-neutral; only skeleton data moves next.

v3 trains **1739851** (C0) / **1739853** (C2) still running — decode A/B after they finish.

### Updates — production hardening H2/M3/M1 + tests (2026-09-10 ~14:25)

| Item | Change |
|------|--------|
| **H2** | Removed dead Nemotron fingerprint/tokenize duplicates from `dataset_cache.py`; fingerprint is only `conversion_baseline.nemotron_cache_fingerprint` (re-exported by loaders). |
| **M3** | Fingerprint includes ChatML body hash; preprocess version **`v3.1-tplhash`**. (Running v3 jobs keep their already-built cache; new submits rebuild.) |
| **M1** | Export writes `tokenizer_config.chat_template` from Hub `chat_template.jinja`. |
| **Tests** | `tests/test_pipeline_hardening.py` — fingerprint identity, template-hash sensitivity, EMA export asserts, seq-extend restore contract. |

Do **not** cancel in-flight **1739851/1739853** for v3.1 — they already materialized v3-richmath cache.

### Updates — v3 trains done + decode A/B queued (2026-09-11)

| Train | Status | Ckpt |
|-------|--------|------|
| C0 **1739851** | COMPLETED | `…/ar2block_masked_1739851/checkpoints/last.ckpt` |
| C2 **1739853** | COMPLETED | `…/ar2block_masked_1739853/checkpoints/last.ckpt` |

Cross-stack **1744042** (C2eb `1735919` → Hub generate): GSM8K flex **38.3%** / IFEval **19.0%** ≈ our BlockSampler ~40% → gap is **data/weights**, not decoder.

Decode A/B submitted (gsm8k+ifeval, thr=1, 8×4, `EVAL_MAX_SEQ_LEN=8192`):

| Job | Ckpt | Profile | OUT |
|-----|------|---------|-----|
| **1760346** | C0 `1739851` | hubmatch | `…/1739851/lm_eval_m2048_hubmatch_t1` |
| **1760347** | C0 `1739851` | dual_cache | `…/1739851/lm_eval_m2048_dual_cache_t1` |
| **1760348** | C2 `1739853` | hubmatch | `…/1739853/lm_eval_m2048_hubmatch_t1` |
| **1760349** | C2 `1739853` | dual_cache | `…/1739853/lm_eval_m2048_dual_cache_t1` |

### Updates — v3 A/B results + Cell B longer-train (2026-09-11 ~22:05)

| Model | Profile | GSM8K flex | IFEval | Notes |
|-------|---------|------------|--------|-------|
| C0 `1739851` | hubmatch | **45.1%** | 23.3% | 1760346 |
| C0 `1739851` | dual_cache | (was cancelled) | — | **Resubmit 1762311** |
| C2 `1739853` | hubmatch | **34.5%** | 26.4% | 1760348 |
| C2 `1739853` | dual_cache | **33.4%** | 25.5% | 1760349 |

v3 richer math did **not** lift C2 (below old C2eb ~40%); C0 > C2 on GSM8K. Cross-stack already said generate ≠ 23 pp gap.

**Data-grid Cell B** (same mix, more tokens): resume both arms **6000 → 12000** steps:

| Job | Arm | Run |
|-----|-----|-----|
| **1762325** | C0 | resume `ar2block_masked_1739851` |
| **1762329** | C2 | resume `ar2block_masked_1739853` |

After done: hubmatch-only lm-eval both. If still ≪50% → Cell A (heavier math) or C (new public math mix).

### Updates — PLAIN-CE-DENOM fix + C2 retrain (2026-09-11 ~22:25)

**Root cause (Hub re-audit):** Fast-dLLM `hub_ref/modeling.py` sets `labels=-100` on clean sites, then HF `ForCausalLMLoss` **means over mask labels only**. Our `plain_ce` zeroed clean CE in the numerator but `_loss` divided by `valid_tokens.sum()`. With complementary fused 2B, valid is doubled → effective mean ≈ **½ Hub** at the same LR → C2 under-updates vs C0 ELBO and vs Hub.

**Fix landed:**
- `BlockTrainer.nll` records mask-site indicators; `_loss` uses `_plain_ce_token_count` when `loss_weighting=plain_ce`
- Regression tests: `test_plain_ce_num_tokens_is_mask_sites_only`, `test_plain_ce_complementary_denom_is_mask_union_not_2b_valid`
- Docs: `LEVERS.md` **PLAIN-CE-DENOM**, registry `loss_plain_ce` notes

**Jobs:**
| Action | Job | Notes |
|--------|-----|-------|
| **Cancelled** | **1762329** | C2 Cell B resume on buggy denom — do not trust |
| **Kept** | **1762325** | C0 Cell B (ELBO; denom bug N/A) |
| **Kept** | **1762311** | C0 DualCache eval |
| **Submitted** | **1762504** | Fresh `C2_fdllm_full` @6000 with fix (same caps; preprocess `v3.1-tplhash`) |

Invalidate interpreting C2 `1739853` / resume `1762329` as Fast-dLLM fidelity. Next: hubmatch lm-eval **1762504** vs C0 `1739851` (~45%).

**Also cancelled C0 Cell B `1762325`** — 12k not needed until fixed C2@6k is gated; keep GPUs for **1762504**.

### Updates — re-audit: SFT-ATTN-PROMPT (2026-09-11 ~22:35)

Full Hub re-read found another **critical** wiring bug (independent of plain_ce):

| Bug | Effect |
|-----|--------|
| **SFT-ATTN-PROMPT** | `nll` used assistant-only `valid_tokens` as SDPA `attention_mask` → **prompt invisible** during train. Hub only overwrites attn with structural block-diff mask; prompt stays visible; `labels=-100` drops prompt from CE only. |
| **PLAIN-CE-DENOM** | (prior) mask-site mean — still required for C2 |

Hits **both** C0 and C2 (absolute Hub gap). Does **not** by itself explain C2≪C0 (both arms shared the attn bug).

**Jobs:**
| Action | Job |
|--------|-----|
| Cancelled | **1762504** (denom-only C2; missing attn fix) |
| Submitted | **1762534** C0 @6000 (attn fix) |
| Submitted | **1762536** C2_fdllm_full @6000 (attn + plain_ce denom) |

Fair gate = hubmatch lm-eval **1762534** vs **1762536**. Invalidate prior C0 `1739851` / C2 `1739853` / `1762504` for Hub-fidelity claims.

### Updates — audit follow-up (2026-09-11 ~22:40)

[Deep Hub vs ours audit](d3356f81-abdc-4945-bae0-bd2b253ad21e) residual actions:

| Item | Action |
|------|--------|
| WT vs HEAD deploy | Confirmed: sbatch uses `PYTHONPATH=${REPO_ROOT}/src` — jobs **1762534/1762536** load working-tree denom+attn fixes (not stale HEAD). |
| **fast_dllm double-eps** | **Fixed in WT**: `mask_schedule=fast_dllm` samples `t~U(0,1)` (no `sampling_eps` floor) then `p=(1-ε)t+ε`. C2 job **1762536** started ~3 min before this patch — **mild** schedule bias unless restarted. |
| t-bucket `plain_ce` denom | Fixed (diagnostic only). |
| Shared LR ELBO vs CE (~7×) | Not a wiring bug; interpret C0 vs C2 loss curves carefully; optional later LR retune. |

### Why Hub-in-tree still left bugs (process failure)

Hub being under `third_party/Fast-dLLM/v2` is **not** the same as parity. Failure mode:

1. **Read-and-declare-match** instead of **numerical lock** against Hub’s observable loss.
2. **Forked stack** (Lightning + our Qwen wrapper) — every glue point (attn mask arg, denom, shift) can diverge even when `modeling.py` is right.
3. **Open train script ≠ 1.5B recipe** — `finetune_alpaca.sh` is LMFlow/Alpaca demo; released GSM8K~62% weights used ~1B-token recipe **not fully disclosed** in that script. `hub_ref/modeling.py` is the real train-objective source of truth.
4. Fixes lived in **working tree** while `git HEAD` still had denom bug — easy to reintroduce.

**Lock added:** `tests/test_hub_plain_ce_parity.py` — our `plain_ce` (+ complementary) must match Hub label=`-100` + shifted CE **mean** on the same tensors. That class of bug should not pass CI again.

### Updates — residual HARD train diffs from audit (2026-09-11 ~22:45)

From [C2 vs Hub residual diffs](b1a7c785-1d8c-4b11-8fd6-73c9950e6267):

| Fix | Change |
|-----|--------|
| **hub_struct_attn_only** | Hub overwrites attn with structural mask only (pads visible). New lever + on `C2_fdllm_full` / `fastdllm_v2`. |
| **ignore_bos=false** | Same lever (Hub has no BOS force-clean). |
| **antithetic off under fast_dllm** | Code: i.i.d. `t` like Hub. |
| **decode thr=1 + no ban MASK** | `fdllm_confidence_decode` + hubmatch/dual_cache profiles. |

**Running jobs 1762534/1762536 do NOT include these** (started earlier). Need explicit restart of **C2** (and ideally C0 only if comparing apples-to-apples on ignore_bos — C0 keeps defaults). Ask before `scancel`.

### Updates — C2 full restart (2026-09-11 ~22:47)

| Action | Job |
|--------|-----|
| Cancelled | **1762536** (pre-`hub_train_parity`) |
| Submitted | **1762670** `C2_fdllm_full` @6000 with shift+comp+fdllm+plain_ce+**hub_train_parity** (`hub_struct_attn_only`, `ignore_bos=false`) |
| Kept | **1762534** C0 (skeleton; Hub pad-attn / ignore_bos N/A) |

Fair gate: hubmatch **1762534** vs **1762670**.

### Updates — C2 Hydra fail + resubmit (2026-09-11 ~23:55)

**1762670 FAILED** (~4 min): `algo.hub_struct_attn_only` missing from `block_masked.yaml` (only in unused `_block_hooks.yaml` comment). Fixed by adding the key to `block_masked` / `uniform` / `hybrid`.

| Job | Status |
|-----|--------|
| **1762670** | FAILED (Hydra) |
| **1763301** | RUNNING — C2 with full hub_train_parity |
| **1762534** | RUNNING — C0 (~1.2h) |

### Updates — trains done + lm-eval queued (2026-09-12 ~12:37)

| Train | Status | Ckpt |
|-------|--------|------|
| C0 **1762534** | COMPLETED | `…/ar2block_masked_1762534/checkpoints/last.ckpt` |
| C2 **1763301** | COMPLETED | `…/ar2block_masked_1763301/checkpoints/last.ckpt` |

Decode A/B submitted (`gsm8k,ifeval`, thr=1, 8×4, `EVAL_MAX_SEQ_LEN=8192`):

| Job | Ckpt | Profile | OUT |
|-----|------|---------|-----|
| **1766642** | C0 `1762534` | hubmatch | `…/1762534/lm_eval_m2048_hubmatch_t1` |
| **1766643** | C0 `1762534` | dual_cache | `…/1762534/lm_eval_m2048_dual_cache_t1` |
| **1766644** | C2 `1763301` | hubmatch | `…/1763301/lm_eval_m2048_hubmatch_t1` |
| **1766645** | C2 `1763301` | dual_cache | `…/1763301/lm_eval_m2048_dual_cache_t1` |

Compare to stale v3: C0 hubmatch ~45% / C2 ~34%. Fair gate = **1766642** vs **1766644**.

### Updates — fixed C0/C2 lm-eval results (2026-09-12 ~13:40)

| Model | Profile | GSM8K flex | IFEval | Job |
|-------|---------|------------|--------|-----|
| C0 `1762534` | hubmatch | **62.1%** | 20.5% | 1766642 |
| C0 `1762534` | dual_cache | **64.5%** | 21.8% | 1766643 |
| C2 `1763301` | hubmatch | **66.8%** | 22.4% | 1766644 |
| C2 `1763301` | dual_cache | **65.9%** | 22.0% | 1766645 |

**Verdict:** Wiring fixes worked. **C2 ≥ C0** on GSM8K (was C2≪C0 ~34% vs ~45%). Absolute GSM8K is at/above Hub 1.5B (~62%). IFEval still well below Hub (~47%) — remaining gap is likely data/instruction mix, not decode/levers.

### Updates — Hub table fill-in queued (2026-09-12 ~13:45)

hubmatch thr=1, `TASKS=mmlu,humaneval,humaneval_plus,mbpp,mbpp_plus` (gsm8k/ifeval already done):

| Job | Ckpt | OUT |
|-----|------|-----|
| **1767239** | C0 `1762534` | `…/1762534/lm_eval_m2048_hubmatch_t1_mmlu_code` |
| **1767240** | C2 `1763301` | `…/1763301/lm_eval_m2048_hubmatch_t1_mmlu_code` |

After: assemble vs Hub 1.5B (MMLU~55, HumanEval~44/40, MBPP~50/41, IFEval~47).

### Updates — Hub mix Pareto + merge + axis expansion (2026-09-16)

**Status before today:** C2 recipe works; Hub gaps remain on MMLU/IFEval. Row-mix search mapped a **chat↔math Pareto front** (not a single winning mix). Fast-dLLM publishes LLaMA-Nemotron post-training **subset** without ratios (~3.15B tokens @ 1.5B from step math).

| Ref | Mix | MMLU | GSM8K | IFEval |
|-----|-----|------|-------|--------|
| Hub 1.5B | (undisclosed subset) | 53.5 | 62.8 | 44.4 |
| Chat-only C2 `1773298` | chat,safety,science | **40.3** | 43.4 | 23.8 |
| Math1M/code500k C2 `1763301` | +math/code caps | 30.6 | **66.8** | 22.4 |
| Mix sweeps | various | ~29–32 | ~57–62 | ~21–24 |

**Decision:** Stop further row-mix grids. Treat Axis-1 as “recipe works; quality mix-limited under unpublished ratio.” Expand transformation space (NLD / uniform / hybrid / native) while waiting on queue.

#### Checkpoint merges (chat × math)

Linear merge `1773298` × `1763301` at α∈{0.3,0.5,0.7} →
`Diffusion/outputs/block_qwen/merge_C2_chat1773298_math1763301_a{30,50,70}/`.

| Job | α | Notes |
|-----|---|-------|
| 1826351–53 | 0.3/0.5/0.7 | **CANCELLED by 0** (~2 min) — node prologue/health (`jpbo-030-47` drained) |
| **1836666–68** | 0.3/0.5/0.7 | Resubmitted hubmatch `mmlu,gsm8k,ifeval` 8×4 — **PENDING** |

#### Explorative trains queued 2026-09-16 (chat/safety/science unless noted)

| Job | Cell | Nodes | Scale |
|-----|------|-------|-------|
| **1836723** | C5_causal_clean (masked, NLD joint) | 8 | paper |
| **1836725** | C2+C5 (fdllm + joint AR) | 8 | paper |
| **1836729** | B1 uniform **C0-style** (`neutral`, no BlockGen levers) | 8 | paper |
| **1836776** | C5 joint on **uniform** | 8 | paper |
| **1836795** | N0 native uniform | 8 | paper |
| **1836796** | Native masked floor (`neutral` `line=block`) | 8 | paper |
| **1836800** | Continue-FT chat`1773298`→math1M/code500k, max_steps=7500 | 8 | paper |
| **1836811** | B4_hybrid_p10 | 8 | paper |
| **1836802** | C5 uniform | 1 | micro |
| **1836803** | N0 uniform | 1 | micro |
| **1836812** | B4 hybrid | 1 | micro |

**Ops:** Fixed stale `ar2block_hybrid.sbatch` (old uni-d2 path / no `--account=scifi`) so B4 submits on booster.

**Git:** Local commit `22dc66b` (Hub eval + C5/merge tooling) — **push blocked** (no GitHub credentials on login node). Branch `ar-to-diffusion-qwen` ahead 1.

**Not updated elsewhere:** `PAPER_EXPERIMENTS.md` / `LEVERS.md` still describe cells; this log is the live job ledger.


### Updates — uniform/hybrid MMLU eval fix (2026-09-17)

**Root cause:** auto-eval after train forced `FORCE_STACK=fastdllm_lm_eval` +
`paper_acc` (likelihood `mmlu`) onto uniform/hybrid ckpts. Masked first-token
CE loglikelihood is invalid for Unif(V)/hybrid (`require_masked_likelihood`).

**Literature:** BlockGen reports generative GSM8K + ELBO/GenPPL (no MMLU LL).
LLaDA-Instruct uses conditional generation for MCQ. Duo uses a USDM likelihood
bound — not our masked heuristic.

**Fix:** (1) launch honors arm — masked keeps hubmatch `paper_acc`; uniform/hybrid
route `blockgen_arpc`. (2) hybrid classified like uniform. (3) new suite
`paper_gen` = `mmlu_generative,gsm8k,ifeval`. (4) `blockgen_arpc` submits
offline + ARPC + generative lm-eval.

**Relaunch:** offline/ARPC/gen for `1836729`, `1836811`, `1836795`, `1836776`
→ jobs `1848191`–`1848202`.

### Updates — overnight wave + uniform C0/C2 + BlockGen transfer (2026-09-17 ~08:05)

#### Merge evals (chat×math linear; hubmatch) — DONE

| α | Job | MMLU | GSM8K | IFEval |
|---|-----|------|-------|--------|
| 0.3 | `1836666` | 38.5 | **64.5** | 22.9 |
| 0.5 | `1836667` | 39.3 | 59.9 | 21.1 |
| 0.7 | `1836668` | **41.4** | 55.0 | **24.6** |

Still on chat↔math Pareto; IFEval ~21–25 (Hub ~44). No three-way Hub close.

#### Explorative trains — DONE + scored (hubmatch where available)

| Job | Cell | Mix (as run) | MMLU | GSM | IFEval | Notes |
|-----|------|--------------|------|-----|--------|-------|
| `1836723` | C5_causal_clean masked | chat/safety/science | 42.0 | 38.5 | 25.0 | best IF among ours |
| `1836725` | C2+C5 masked | (fdllm+joint) | 29.8 | 44.0 | 21.1 | |
| `1836800` | continue-FT chat→math | math1M/code500k | 38.1 | 59.8 | 23.3 | did not beat merge α=0.3 GSM |
| `1836796` | native masked floor | | 23.0 | 0.5 | 8.1 | scratch floor |
| `1836729` | B1 uniform neutral | **chat-only** (mismatched vs C0) | — | — | — | gen eval pending |
| `1836776` | joint_ar+causal_clean uniform | chat-only | — | — | — | gen eval pending |
| `1836795` | N0 native uniform | | — | — | — | gen eval pending |
| `1836811` | B4 hybrid p10 | | — | — | — | gen eval pending; no ARPC |

First auto lm-eval on uniform/hybrid (`1841567/83/205/2682`) **FAILED** on likelihood MMLU — see prior fix note.

#### paper_gen relaunch (2026-09-17)

| Job | Role | State (~08:05) |
|-----|------|----------------|
| `1848191/94/97/200` | offline eval | **COMPLETED** |
| `1848192/95/98/201` | ARPC | **FAILED** — Hydra struct: ckpt cfg missing `sampling.use_arpc`; need `+sampling.…` |
| `1848193/96/99/202` | `paper_gen` lm-eval | **RUNNING** (~1.6h, on `mmlu_generative`) |

**ARPC fix:** `scripts/slurm/arpc_decode_eval.sbatch` now uses `+sampling.use_arpc=true` (etc.). Resubmitted uniform-only ARPC: `1849336`–`1849338` (skipped hybrid).

#### Best mix decision + uniform C0 / recipe+ / BlockGen transfer

**Mix for matched uniform spine:** same as masked C0/C2-math —  
`NEMOTRON_SFT_SPLITS=chat,safety,science,math,code` (math1M / code500k defaults).

**Constraint:** Fast-dLLM C2 train levers are **masked-only** (`BlockTrainer`). Uniform “C2” ≠ fdllm hooks.

| Preset | Meaning | Job | State |
|--------|---------|-----|-------|
| **U0** | uniform ≈ C0 (hooks off, math mix) | `1848814` | **FAILED** CUDA node death exit 9 (~6m) |
| **U0** retry | same | **`1849335`** | **RUNNING** |
| **U2** | uniform recipe+ = `joint_ar`+`causal_clean` (NLD-style; **not** BlockGen) | **`1848844`** | **RUNNING** (~42m) |
| **xfer_blockgen_uniform_32** | AR→uniform + official BlockGen hook stack (1+32 5/95, u-strat, pure_noise@1, CE@1). Train hydra baked **`arpc_corruption=divergence`** (pre-fix registry). Decode must force **`ar_metric`/`nll`** (`eval_arpc_armetric`). **Transfer**, not native BlockGen. Gap: no `x0_causal`. | **`1849287`** | **DONE** |
| duplicates `1848871/73` | U0/U2 double-submit | CANCELLED | |

#### Absorb ARPC (masked BlockGen equivalent) — enabled 2026-09-17

BlockGen ARPC on absorb is now first-class (was uniform-only refuse):

- Sampler: masked guided steps remask lowest-LL tokens with `mask_id`
- Pins match TinyGSM scripts: `arpc_mode=blockgen`, `arpc_corruption_mode=ar_metric`, `arpc_ar_metric=nll`
- Lever `arpc_blockgen` / presets `B3_arpc`, `xfer_arpc`, **`xfer_bg_mix_32_arpc`** allow `arms: [masked, uniform]`
- Decode: `scripts/slurm/arpc_decode_eval.sbatch` (forces `unmask_threshold=null`)
- Prerequisite: size-1 in train mix (`xfer_bg_mix_32` jobs **1855537** / **1855541** qualify once done)

#### C0/U0 skeleton + BlockGen mixture (fair cross-arm)

Same train levers on **both** arms (no ARPC — masked-compatible). Math mix. Not renaming C0/U0.

| Preset | Arm | Meaning | Job | State |
|--------|-----|---------|-----|-------|
| **xfer_bg_mix_32** | masked | C0 skeleton + 1+32 @5/95 + u-strat + pure_noise@1 + CE@1 | **`1855537`** | **PENDING** |
| **xfer_bg_mix_32** | uniform | U0 skeleton + same mix (pair of above; no ARPC pin) | **`1855541`** | **PENDING** |
| **xfer_bg_mix_32_arpc** | masked/uniform | same mix + absorb/uniform ARPC decode pins | — | use after mix train, or launch fresh |

Overrides: `block_weights=[0.05,…,0.95]`, `u-stratified`, `pure_noise_block_sizes=[1]`, `loss_type_special_cases=[1,ce]`; `BLOCK=32`, paper 6k/256. Tag=`xfer_bg_mix_32`.

Registry adds: `U0`, `U2`, `xfer_bg_mix_32`, `xfer_bg_mix_32_arpc`, `xfer_blockgen_uniform` (1+16, need `BLOCK=16`), `xfer_blockgen_uniform_32`.

Audited upstream: `third_party/blockgen` ([jdeschena/blockgen](https://github.com/jdeschena/blockgen)); OWT script `scripts/train/owt/blockgen_uniform_1_16.sh`.

**Thesis naming:** Conv-Base/Conv-Full for masked; U0/U2/xfer for uniform. Old chat-only uniform B1 is appendix-only (mix mismatch).

**Still open:** C3 AR SFT; Tab 2 lever ablation; Hub final ratios; paper_gen SUMMARYs; U0/U2/xfer + xfer_bg_mix scores; absorb ARPC eval on 1855537.

### Fix — fake free-gen collapse + ARPC spam (2026-09-17 ~13:55)

**Not model collapse.** `conversion_free` prefixes contain `<|im_end|>` after the
system turn. `_truncate_im_end` and gen-PPL `first_chunk_only` treated that as
generation EOS → `samples.txt` looked empty and metrics showed eos_rate=1.0 /
PPL≈402 with honesty_warning.

**Code:** prefix-aware truncate (`generate_samples.py`) + prefix_len-aware EOS
trim/stats (`generative_ppl.py`). ARPC submit gated on `EVAL_HAS_ARPC_SIZE1`
(`submit_family_eval` / `_infer_block_qwen_eval_profile.bash`); trainer size-1
check is warn-only on decode. Tests in `test_decode_profiles.py` (9 passed).

**Recomputed** U0 `1849335` / xfer `1849287` offline metrics:
| cell | gen-PPL (fixed) | eos_rate | mean toks to EOS |
|------|-----------------|----------|------------------|
| U0 | ~60.8 | 0.17 | ~1766 |
| xfer | ~55.2 | 0.09 | ~1882 |

ARPC on U0/U2/chat-only uniforms **correctly skipped** (no size-1). Keep ARPC
for `xfer_*` / `xfer_bg_mix_*` only. Re-decode U2 `1848844` after its offline
eval finishes (same fix).

### Hygiene — unigram entropy beside GenPPL (2026-09-17)

Unifusion-style: `generative_ppl` now writes sample-mean Shannon unigram H
(nats, eval tokenizer, prefix stripped) + flags `low_ppl_low_unigram_entropy`
when PPL&lt;80 and H&lt;4.5. Tests in `test_decode_profiles.py`. Merged into
existing U0/xfer `gen_ppl_metrics.json` (no model reload):

| cell | gen-PPL | H_mean | H_med | honesty | cite? |
|------|---------|--------|-------|---------|-------|
| U0 | 60.8 | **6.04** | 6.62 | none | hygiene only |
| xfer | 55.2 | **6.25** | 6.60 | none | hygiene only |
| U2 | 86.8 | **6.26** | 6.64 | none | hygiene only |

**Collapse gate cleared ≠ fluency.** Decodes are word-salad / soup under
`conversion_free`; `full_length_rate`≈0.83–0.91. Cite as
`(PPL, H)` hygiene only — never as “fluent.” Ranking
`xfer < U0 < U2` is confounded (xfer has BlockGen train geometry).

### Submitted — thesis-critical C3 + Tab 2 (2026-09-17 ~13:58)

Fixed stale `ar_sft.sbatch` (was `/fast` + 2-GPU standard) → Jupiter booster
8×4, math mix. `submit_paper_cell.sh` now pins math1M/code500k for C3 + lever cells.

| Cell | Job | Mix | Notes |
|------|-----|-----|-------|
| **C3** AR SFT | **`1856100`** | chat+safety+science+math1M+code500k | RQ2 / Tab 1 |
| **C2_shift** | **`1856101`** | same | Tab 2 |
| **C2_comp** | **`1856103`** | same | Tab 2 (warn: comp alone) |
| **C2_fdllm** | **`1856105`** | same | Tab 2 strict shift+comp |

U2 offline `1854401` still running; watcher re-decodes + gen-PPL when done
(`slurm_logs/u2_redecode_watch.log`).

### Fix + resubmit — C3 `1856100` FAILED (2026-09-17)

**Cause:** `ar_sft.sbatch` used plain `srun` without restoring
`CUDA_VISIBLE_DEVICES=0,1,2,3`. Slurm prolog sets per-task CVD=`[0]` → Lightning
`trainer.devices=4` → `MisconfigurationException: requested [0,1,2,3] but only [0]`.
Exit ~2.5 min. Block jobs already use `_block_qwen_ddp.bash` (CVD + `srun_retry`).

**Fix:** `ar_sft.sbatch` now sources shared DDP helper (`run_block_qwen_srun_train`).

**Resubmit:** C3 **`1857271`** (fresh `ar_sft_${jid}`, same 8×4 + math mix).

### Go — U0_shift + U2 hygiene (2026-09-17 ~15:40)

| Item | Result |
|------|--------|
| **U0_shift** | Submitted **`1857278`** (`shift_loss_targets`, math mix, paper 8×4) |
| **U2** `1848844` | Watcher redecode died (`f-string` SyntaxError). Offline metrics already OK (eos_rate=0.125, PPL≈86.8). Rewrote `samples.txt` prefix-aware + merged unigram H (**6.26**, no honesty flag). |
| C3 | still PENDING **`1857271`** |
| Tab 2 | `1856101/03/05` RUNNING |

### Unifusion novelty gaps closed (isolated; 2026-09-17 ~16:05)

**Safety:** defaults remain off (`intra_block_attn_anneal_steps=0`,
`kernel_anneal_steps=0`, `shift_on_hybrid=false`, `track_revisions=false`).
Canonical C0/C2/U0 recipes and run dirs **untouched**. New cells use fresh
`ar2block_*_<jid>` + `WANDB_RESUME=never`.

| # | Gap | Implementation | Job |
|---|-----|----------------|-----|
| 1 | Intra-block causal→bi anneal | `block_mask.intra_block_open` + presets `U0_anneal` / `C0_anneal` | **`1857471`** / **`1857473`** |
| 2+3 | Shared x0 / two-stage | `submit_u0_from_c0.sh` → fresh uniform + C0 resume | **`1857519`** |
| 4 | Revision-rate probe | `sampling.track_revisions` + `revision_probe.sbatch` | **`1857478`** |
| 5 | Hybrid × x0 | `B4_shift_explorative` (`shift_on_hybrid`) | **`1857476`** |
| 6 | Joint curriculum | `B4_joint_curriculum` (anneal + kernel_p ramp) | **`1857475`** |

**Rule:** GenPPL never alone → `(PPL, H_mean[, H_med])` via `tools/gen_ppl_hygiene.py`
+ `scripts/run_gen_ppl_hygiene.sh` / `submit_gen_ppl_hygiene.sh` / cell **C4**.

| Piece | Status |
|-------|--------|
| Pair sidecars + collapse panels | Done offline for U0/xfer/U2 |
| Dual GenPPL + NFE {8,16,32,64} + panel | **C0** `1857349`, **U0** `1857352` (+multiseed), **C2math** `1857356` |
| Optional WinoGrande slice | **C0** `1857350`, **U0** `1857353` (C3 when `1857271` done) |
| Docs | `PAPER_EXPERIMENTS.md` Fig 4 / reporting rule; `cells.yaml` C4 notes |

### Plan add — gated AR→masked→uniform (`U0_from_C0`) (2026-09-17)

Unifusion: direct AR→uniform ≈ two-stage on GenPPL. We still **plan to try**
block-Instruct continue-FT from C0/`1762534` → uniform **only if** direct U0
`1849335` paper_gen is weak vs C0 (see `PAPER_EXPERIMENTS_POPULATED.md` §5
Axis C follow-up). Priority behind C3 + Tab 2. Optional `U0_from_C2` only if
warm-start helps.

### Plan + code — Unifusion-style shift on uniform (`U0_shift`) (2026-09-17)

**Same idea as Unifusion \(x_0\) shift** (logits at i → clean i+1); our hook is
`shift_loss_targets` (Fast-dLLM). Was masked-only; now
`LOSS_SPECIAL_CASE_POLICY=masked_and_uniform`. Hybrid still refuses.

| Cell | Job / status |
|------|----------------|
| C0 vs **C0_shift**/C2_shift | C2_shift train **`1856101`** |
| U0 vs **U0_shift** | Planned; `./scripts/submit_paper_cell.sh U0_shift` |

See populated §5 Axis D follow-up.

### Ops — C3 id + duplicate U0_from_C0 (2026-09-17 ~16:42)

- C3 `1857271` CANCELLED → replacement **`1857323`** (PENDING; same CVD-fixed `ar_sft.sbatch`).
- Duplicate U0_from_C0: cancelled **`1857518`** (Diffusion path resume); keep **`1857519`** (Diffusion-new C0 `last.ckpt`, max_steps=7500).

### Fix — GenPPL / ARPC deep-audit (2026-09-17 ~17:15)

**What was wrong**
1. Status line “fluency OK” overclaimed collapse-gate pass as linguistic fluency.
2. xfer ARPC GenPPL≈402 (`eval_arpc`, n=8) = **prefix-EOS bug** (first EOS at
   chat `<|im_end|>`); `samples.txt` truncated; tensor still had long gens.
3. That ARPC job (`1853871`) used **`corruption=divergence`**; TinyGSM pin is
   **`ar_metric`/`nll`**. Train `1849287` hydra also baked divergence (registry
   still had that default at launch; working tree now `ar_metric`).
4. U2 `block_elbo_sweep` **size-1 PPL≈459** vs U0/xfer ≈3.5 — do not cite size-1
   until joint_ar/causal_clean ELBO path audited (4/16/32 look normal).

**Fixes shipped**
- Default `sampling.arpc_corruption_mode=ar_metric` (`configs/sampling/block.yaml`,
  `BlockSampler`, registry `arpc_blockgen`, `submit_family_eval` exports).
- GenPPL: `citeable=false` on too-few-tokens / short-span honesty; `fluency_note`
  on high `full_length_rate`; hygiene `pair_from_metrics` refuses unciteable.
- Legacy ARPC → `eval_arpc_legacy_divergence_prefixbug/`; `samples.txt` rewritten
  prefix-aware; metrics stubbed unciteable.
- Fresh ARPC decode **`1858745`** → `eval_arpc_armetric/` (n=64, ar_metric/nll).
  Result: **(82.7, H̄=6.35)** citeable as hygiene; **worse** than ancestral baseline
  55.2 — do not claim ARPC helps GenPPL on this convert xfer ckpt. Still soup
  (`full_length_rate`≈0.98).

**Cite rule:** baseline `eval/` (PPL,H) hygiene only; **never** cite ARPC≈402;
use `eval_arpc_armetric` (82.7) if comparing ARPC at all.

**Still true (not fixed by the above code):**
1. Free-gen under `conversion_free` is still **soup**.
2. **xfer ≠ U0 recipe** — do not rank/interpret as matched uniform.

### Fix — U2 size-1 ELBO meter (2026-09-18)

**Root cause:** `block_elbo_sweep` / `BlockTrainer.nll(train_mode=False)` scored
size-1 with continuous-time **ELBO** over \(t\sim U(0,1)\). BlockGen eval
(`third_party/blockgen/algo.py`) uses **CE + pure noise (t=1)** at size-1.
Joint-AR/causal_clean models (U2) are over-confident at low-t under size-1
attention → ELBO terms explode (PPL≈459). Size 4/16/32 ELBO was always fine.

**Audit:** U2 vs U0 fixed-t probe — size-1 t=0.1 NLL 8.5 vs 1.6; size-32 t=0.1
both healthy. Zeroing `joint_ar_alpha` at eval did not change diffusion NLL
(path was already diffusion-only).

**Fix:** eval `block_size==1` → CE + `t=1` (training unchanged). Sweep annotates
`meter=ce_pure_noise`. Unit test `test_size1_eval_uses_ce_not_elbo`.

**Re-sweep (30 val batches, BlockGen size-1 meter):**

| cell | size-1 (CE+t=1) | size-4 ELBO | size-32 ELBO |
|------|-----------------|-------------|--------------|
| U2 `1848844` | **9.89** (was 459) | 4.13 | 5.73 |
| U0 `1849335` | **3.76** | 4.37 | 5.34 |

Cite size-1 rows again under the CE meter; keep free-gen soup / xfer≠U0 as STILL-TRUE.

### Deep-dive — STILL-TRUE soup + xfer≠U0 (2026-09-18)

Full writeup: `docs/research/DESIGN_LOCKS.md` § Deep-dive — STILL-TRUE.

**Soup:** U0/xfer/U2/C0 open free-gen all `soup_not_fluent` (FLR 0.83–0.91, H̄ healthy).
Primary cause = open-header OOD meter; ARPC/NFE/prefix fixes failed as fluency.
**Action:** demote open free-gen; lead with conditional Instruct.

**xfer≠U0:** `1849287` = 1+32@5/95 + u-strat + pure_noise + CE@1 + ARPC bake;
U0 levers empty. GenPPL 55&lt;61 confounded; mix_u without ARPC is **66&gt;61**.
**Action:** xfer only in RQ-B2/X tables; never claim K / matched U0 floor.

### Fix — paper_gen 12h TIMEOUT on mmlu_generative (2026-09-18)

**Cause:** `MAX_NEW=2048` paid in full by BlockSampler; task `until` stops only
truncate **after** decode. MMLU-gen never finished in 12h.

**Fix:** `cap_max_new_for_task` ceilings — mmlu_generative→64, gsm8k→512,
ifeval→1024 (`block_qwen_eval_utils.py` + `generate_until`).

**Ops:** cancelled hung `1862499/3036/3135`. Resubmitted paper_gen →
`lm_eval_paper_gen_cap/`:
| cell | job |
|------|-----|
| U0 `1849335` | **1870627** |
| xfer `1849287` | **1870628** |
| U2 `1848844` | **1870629** |
| xfer_mix_u `1855541` | **1870630** |
| U0_shift `1857278` | **1870631** |
| U0_anneal `1857471` | **1870632** |
| U0_from_C0 `1857519` | **1870633** |
| B4_joint `1857475` | **1870634** |

### Hub Fast-dLLM × C2 mix correctness check (2026-09-22)

External judge for masked Fast-dLLM pipeline: Hub LMFlow train on our C2
Nemotron mix (math≤1M, code≤500k, chat/safety/science) from Qwen→Fast_dLLM
init, then Hub `eval.py` / lm-eval.

| Piece | Path / job |
|-------|------------|
| Init | `/e/scratch/scifi/elsayed3/hub_fastdllm_c2/init_qwen15b` (Qwen weights → Hub arch) |
| Data | `…/data/c2_mix` — 2,280,138 conversations (~12G JSON) |
| Train | **`1951718`** `hub-fdllm-c2-tr` (8×4, 6k steps, GBS 256, L=2048) |
| Eval (afterok) | **`1951719`** `hub-fdllm-c2-ev` → Hub lm-eval mmlu/gsm8k/ifeval thr=1 |
| Scripts | `scripts/submit_hub_fastdllm_c2.sh`, `scripts/slurm/hub_fastdllm_c2_train.sbatch` |

Compare Hub-trained-on-C2 GSM vs our C2 `1763301` (~67) vs public Hub 1.5B (~62–63).

### U0 hierarchical GSM canary `1950105` — COMPLETED (2026-09-22)

Locked protocol: `DECODE_PROFILE=hierarchical`, `FORCE_GREEDY=0`, ckpt `1849335`.

| Metric | Score |
|--------|-------|
| GSM flexible-extract | **6.37%** (±0.67) |
| GSM strict-match | **0.0%** |

**Verdict:** hierarchical alone does **not** lift U0 off ~6% chance-range. Do **not** requeue full U-family on hierarchy hope. Residual is train/recipe (or deeper decode), not the old baseline+greedy protocol bug.
Out: `…/ar2block_uniform_1849335/lm_eval_hier_gsm_canary_20260922/`
**Deep audit:** `docs/research/DEEP_AUDIT_U0_CANARY_2026-09-22.md` — hierarchical engaged;
residual = fluent wrong CoT / ancestral uniform science (not protocol). Next: C0 ancestral
fairness control; optional ARPC/low-T / single_stream smoke; retrain under V_eff lock.


