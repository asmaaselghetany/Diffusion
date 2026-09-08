# Day log — Collapse forensics, plain_ce sign fix, Fast-dLLM-aligned multi-node eval

**Date:** 2026-09-06  
**Site:** Jülich Supercomputing Centre, Jupiter **booster** (GH200)  
**Working tree:** `/e/project1/scifi/elsayed3/Diffusion-new`  
**Family in scope:** **conversion** — `LINE=ar2block`, masked forward process, Qwen2.5-1.5B-Instruct init, Nemotron post-training SFT packs (`data=sft_qwen` / cache under `/e/scratch/scifi/elsayed3/.cache/discrete_diffusion/block_qwen_sft_nemotron/`)  
**Parent recreate program:** [2026-09-04_reference_recreation](../2026-09-04_reference_recreation/EXPERIMENT.md)  
**Living protocol:** [2026-08-31_postfix/RUN_PROTOCOL.md](../2026-08-31_postfix/RUN_PROTOCOL.md)  
**Design locks:** conversion vs native vs `xfer_*` transfer — never relabel BlockGen knobs on `ar2block` as BlockGen.

This document is intentionally long. It is meant to be mined later for a thesis
Experiments / Failure-analysis chapter without reopening chat transcripts.

---

## 1. Where we stood at the start of the day

The September 4 reference-recreation program had already locked three scientific
families and submitted (or discarded) a set of cells. By the morning of September 6
the conversion story looked internally inconsistent in a way that would be dangerous
to paper over:

**C0 — neutral conversion floor (Slurm train job `1660579`).**  
Run root: `/e/project1/scifi/elsayed3/Diffusion-new/outputs/block_qwen/ar2block_masked_1660579/`  
(checkpoint files are also reachable via the `Diffusion/outputs/...` symlink that
`readlink -f` often resolves to). Training used hooks-off defaults: no
`shift_loss_targets`, no complementary masks, `loss_weighting=elbo`,
`mask_schedule=alpha`, block size 32, sequence length 2048, Instruct init.
Offline eval after train (job family around `1666335`) wrote
`eval/block_elbo_sweep.json`, `eval/gen_ppl_metrics.json`, `eval/samples.pt` /
`samples.txt`. ELBO looked respectable (block-32 mean NLL ≈ 1.82 → BPD ≈ 2.62).
Train progress bars in `slurm_logs/bqwen-C0-masked_1660579.out` show
`trainer/loss` starting near **+11.9** and settling near **+2**. Yet qualitative
free-gen samples were uneven: English chain-of-thought-ish **heads**, messy
**tails**, and gen-PPL ≈ **146** only under `first_chunk_only=true`
(`eval/gen_ppl_metrics.json`: `ppl=146.41`, `tokens=32236`, `pretrained_model=gpt2-large`).

**C2 — Fast-dLLM-ish train-only control (job `1660576`, resumed through `1690702` to step 6000).**  
Run root: `.../outputs/block_qwen/ar2block_masked_1660576/`  
Levers: `algo.shift_loss_targets=true`, `complementary_masks=true`,
`mask_schedule=fast_dllm`, `loss_weighting=plain_ce`. This is the training half of
the Fast-dLLM recipe without forcing Hub confidence decode pins at train time.
lm-eval job `1691158` and live probes showed something much worse than C0 weakness:
nearly every GSM8K continuation collapsed into the repeating Chinese string
**`新人玩家`** (Qwen token id **117294**). That is not “a bit less fluent than C0”;
it is a discrete attractor.

**Poisoned exactness cell.** Job `1692667` (`fastdllm_v2`) was still training with
the same shift+plain_ce objective shape. Its loss trajectory already mirrored C2’s
negative-loss pathology (about **−6 → −135** before cancel). Leaving it up would
have burned four nodes on a known-broken objective.

The day’s scientific obligation was therefore: **separate engineering failure from
recipe failure**, **separate free-gen aesthetics from on-distribution learning**,
and **stop measuring conversion success with metrics Fast-dLLM does not lead with**.

---

## 2. Research questions (stated so they can be falsified)

**RQ-A (C2).** Is the `新人玩家` loop caused by (i) shift-logit mis-alignment at
decode, (ii) a data-frequency mode in Nemotron, (iii) an inverted training
objective, or (iv) something else?

**RQ-B (C0).** Given healthy ELBO and positive loss, is poor long free-gen evidence
that conversion training failed, or evidence that free-gen from bare
`<|im_start|>` is a mismatched probe relative to chat-packed SFT and paper lm-eval?

**RQ-C (Eval).** Can we instrument a Fast-dLLM-v2-shaped reporting surface
(GSM8K, IFEval, HumanEval±, MBPP±, optional MMLU/GPQA/MATH) on multi-node
Accelerate without claiming fused-CUDA throughput parity?

---

## 3. Methods — what we actually ran

### 3.1 Logit and head forensics (C0 vs C2)

Using `discrete_diffusion.evaluations.checkpoint_utils.load_block_trainer_checkpoint`
on GPU, we built a length-2048 canvas of MASK (`mask_id=151665`) with BOS
(`151644`, string `<|im_start|>`) at position 0, then called
`model.backbone_logits(xt, x0)` with `x0=xt` (free-gen dual-stream start). We
compared raw logits to `align_shift_logits` from
`src/discrete_diffusion/sampling/shift_logits.py`. Separately we measured
`lm_head.weight` row norms for token 117294 and global logit statistics.

For C2 we repeated the first-step probe on checkpoints
`0-500`, `0-1000`, `0-2000`, `1-3000`, `2-6000` under
`outputs/block_qwen/ar2block_masked_1660576/checkpoints/`.

### 3.2 Offline free-gen corpus statistics (C0)

Artifact: `outputs/block_qwen/ar2block_masked_1660579/eval/samples.pt` shape
`(64, 2048)`. We decoded with the Qwen tokenizer and measured, by position
bucket, latin / CJK / punct / “other” rates and top token identities. We also
counted how often position 1 was token `766` (`ink`).

### 3.3 Live BlockSampler generations (C0)

`model._create_sampler().generate(...)` with `max_new_tokens` short (128–256),
comparing ancestral vs greedy vs `p_nucleus=0.9`, prompt continuation
(“The sky is blue because”), and `apply_chat_template` math questions. Config
defaults on the C0 checkpoint: `sampling.steps=32`, `p_nucleus=1.0`,
`greedy=false`, `inject_bos=true`, `algo.ignore_bos=true`.

### 3.4 Train-like reconstruction probe (C0)

On a validation pack sequence from
`nemotron-sft-valid_validation_bs2048_wrapped_qwenchat_d1_53cff9af962c.dat`
we (i) corrupted with the model’s `_corrupt` at `t=0.5` and scored argmax
accuracy on MASK sites, and (ii) froze a real 256-token prefix and generated
128 tokens for qualitative comparison to gold continuation.

### 3.5 Loss trajectory mining

Parsed `trainer/loss=` from:

- `slurm_logs/bqwen-C0-masked_1660579.out`
- `slurm_logs/lever_C2_fdllm_full_masked_1660576.out`
- `slurm_logs/lever_fastdllm_v2_masked_1692667.out` (pre-cancel)
- `slurm_logs/lever_fastdllm_v2_masked_1694554.out` (post-fix)

### 3.6 Code fix and regression

Edited `BlockTrainer._masked_loss` in
`src/discrete_diffusion/algorithms/block_trainer.py`. Added
`test_shift_plain_ce_is_minimize_oriented` in
`tests/test_block_shift_loss.py` (asserts confident-wrong CE ≫ confident-right CE
and positive mean CE under shift+plain_ce).

### 3.7 Eval harness expansion

Extended `examples/block_qwen/lm_eval.sh`, `scripts/slurm/lm_eval.sbatch`,
`scripts/prefetch_lm_eval_data.sh`,
`src/discrete_diffusion/evaluations/block_qwen_lm_eval.py`,
`src/discrete_diffusion/evaluations/decode_throughput.py`. Default lm-eval
allocation became **4 nodes × 4 GPUs**, Accelerate data-parallel, one model
replica per GPU (Fast-dLLM `eval.py` pattern), with Hub DualCache / confidence
exposed as optional decode pins that we explicitly label as **algorithmic**
ports, not fused Fast-dLLM CUDA.

---

## 4. Results — C2 collapse (RQ-A)

### 4.1 Decode is not the villain

On the all-MASK+BOS canvas, C2 @ 6000 assigns enormous mass to token 117294 at
essentially every early position (logit ≈ **779**, rank 1). Enabling or disabling
`align_shift_logits` does not change the argmax. Double-aligning does not either.
The same canvas under C0 yields ordinary logit magnitudes (~15) and English-ish
top tokens (e.g. fragments like `<th`, commas, ` the`) — interestingly already
hinting at think-tag vocabulary, but **not** a single locked Chinese token.

Prompt-conditioned first step on C0 for `"The sky is"` was semantically sensible
(` a`, ` blue`, …). On C2 the same prompt still ranked `新人玩家` first with
logit ≈ 779. So the disease is in the **weights**, not in a sampler-only shift bug.

### 4.2 Collapse begins early and intensifies

| C2 checkpoint | Approx. global step | logit[117294] @ pos1 | `lm_head` ‖row‖ | rough logit std on canvas |
|---------------|---------------------|----------------------|-----------------|---------------------------|
| `0-500.ckpt` | 500 | ~213 | 1.53 | ~14 |
| `0-1000.ckpt` | 1000 | ~262 | 1.88 | ~21 |
| `0-2000.ckpt` | 2000 | ~366 | 2.62 | ~41 |
| `1-3000.ckpt` | 3000 | ~469 | 3.34 | ~62 |
| `2-6000.ckpt` | 6000 | ~779 | 5.46 | ~127 |

C0’s `lm_head` row for 117294 sits near norm **1.37** (about rank 39 among rows).
A freshly loaded AR Qwen2.5-1.5B-Instruct head is similar (~1.38). The C2 head row
for 117294 becomes the **maximum** row norm in the matrix by 6k. Hidden-state
scale is also extreme (logits hundreds), so this is not only a single row
drifting in isolation.

Top-k on C2’s first step are almost entirely Chinese gaming phrases
(`游戏副本`, `战士职业`, …). That pattern is consistent with an **optimization
attractor**, not with “the model copied a frequent Nemotron token”: scans of the
first 20 000 packed training sequences in both
`nemotron-sft-train_train_bs2048_wrapped_qwenchat_d1_53cff9af962c.dat` and
`…_wrapped.dat` found **zero** occurrences of id 117294.

### 4.3 The loss sign explains the attractor

Lightning minimizes `trainer/loss`. For C2 that quantity was **negative and
growing in magnitude** (about **−6.4** at the start of
`lever_C2_fdllm_full_masked_1660576.out`, about **−730** near the end of the
logged progress). For C0 the same meter is positive and decreasing
(**+11.9 → ~+2** in `bqwen-C0-masked_1660579.out`).

Reading `BlockTrainer._masked_loss` made the inconsistency obvious. Under
`shift_loss_targets`, the code truncated to a next-token grid and built

```text
ce = −log pθ(x0[i+1] | …)     # ≥ 0
masked_neg_ce = 1_{mask} · (−ce)
```

For ELBO it then multiplies by `dα/(1−α)` with `dα < 0`, which flips the sign
again and yields a minimize-oriented surrogate. For `plain_ce`, however, it
**returned `masked_neg_ce` unchanged**. The non-shift branch correctly calls
`masked_plain_ce_per_token`, which returns **+CE**. So the Fast-dLLM-shaped
conjunction `shift ∧ plain_ce` — exactly C2 / fastdllm_v2 — was training to
**maximize** cross-entropy on masked sites.

That is sufficient to explain logit explosion on wrong tokens: the optimizer is
rewarded for confident errors. A particular vocabulary direction can win and
then self-reinforce across contexts, producing the observed single-string loops
at decode.

**Fix landed in-tree on 2026-09-06:** for shift+plain_ce return
`mask_positions * ce`. ELBO path unchanged. Regression test
`tests/test_block_shift_loss.py::test_shift_plain_ce_is_minimize_oriented`
checks that a confidently wrong distribution yields large positive CE and a
confidently correct one yields near-zero CE.

### 4.4 What we explicitly do *not* conclude from C2

We do **not** conclude that NVIDIA’s Fast-dLLM recipe is unstable, nor that
complementary masks or `fast_dllm` schedules are inherently toxic. We conclude
that **our** implementation of shift+plain_ce was sign-inverted, and that every
fluency or lm-eval number from `1660576` / `1690702` / `1692667` is scientifically
void for recipe comparison. Those runs remain useful only as an engineering
postmortem.

---

## 5. Results — C0 “bad samples” (RQ-B)

### 5.1 Quantitative free-gen degradation is real but length-progressive

From `eval/samples.pt` (64 sequences):

| Position bucket | Approx. latin token fraction | Notes |
|-----------------|------------------------------|-------|
| [0, 64) | 0.83 | English-looking heads |
| [64, 256) | 0.79 | still mostly latin |
| [256, 512) | 0.73 | CJK/other begin rising |
| [512, 1024) | 0.63 | |
| [1024, 2048) | 0.40 | tails; top tokens include `�` byte pieces (ids 242, 97) |

Residual MASK rate after offline decode was **0**. So the sampler finished
unmasking; the problem is **content**, not stuck masks.

**Start-token pathology.** 64/64 sequences begin with BOS `151644`. **45/64**
have position 1 = `766` (`ink`), producing the visible prefix
`<|im_start|>ink>` or similar. In the training packs, tokens immediately after
BOS are overwhelmingly `user` / `assistant` / `system` (tens of thousands of
counts in a 10k-sequence scan); `ink` after BOS count was **0**. Meanwhile
`<think>` tokenizes as `[13708, 766, 29]` (`<th` + `ink` + `>`), and think-tags
appear in a large minority of Nemotron windows (~774/3000 sequences showed
`<think>` / `</think>` in the first 256 decoded tokens of a sample). Live
first-step logits on bare BOS rank **`<th` (13708)** first at positions 1–4.
Interpretation: free-gen from inject-BOS alone asks the model to invent a role
header under a prior soaked in think-tag pieces; parallel block unmasking then
emits **scrambled tag fragments** (`ink>`, `ink<think>`) before falling into
CoT-like English mush.

### 5.2 On-distribution behavior does not match the free-gen horror story

On a real validation chat sequence (system “detailed thinking on”, then a
marketing multiple-choice about share-of-voice):

- At `t=0.5`, argmax accuracy on masked sites was **≈ 0.68** (989 masked
  positions). That is compatible with a model that actually learned denoising.
- Conditioning on the first 256 gold tokens and generating 128 more produced
  on-topic English about SOV / brand awareness (latin fraction high; not a
  Chinese loop). Token-level match to gold was low (~0.06), which is expected
  for open-ended continuation and should not be mistaken for failure of the
  reverse process.

Short free-gen (≤192 new tokens) under greedy or ancestral was rough but
readable English CoT after the broken tag prefix — not C2-style lock. Chat
template prompts such as “What is 2+2?” / “15+27?” eventually emitted the
correct number (`4`, `45`) but then overran with `<|im_end|>` repetition and
think-tag debris. That points to **stopping / formatting discipline**, not
absence of local competence.

### 5.3 Why loss and free-gen “disagree” without either meter being broken

Training optimizes denoising of **corruptions of real packed chats**. Free-gen
optimizes ancestral samples from **near-pure noise** with a **bare BOS** that
never appears alone in the finetune distribution. ELBO/gen-PPL-on-heads can look
fine while long free-gen looks bad. Fast-dLLM’s public tables primarily report
**conditional** task accuracy under chat-like prompts; using bare 2048 free-gen
as the conversion “roof” therefore answers a different question than the papers
pose.

**Operational verdict for C0:** keep checkpoint `1660579` as the hooks-off floor;
do not retrain it for sample cosmetics; judge it with lm-eval; treat free-gen as
a diagnostic of priors and long-horizon ancestral error accumulation.

---

## 6. Results — eval stack and jobs (RQ-C)

### 6.1 What we changed

We aligned the default reporting surface with Fast-dLLM v2’s `eval_script.sh`
spirit: MMLU, GPQA, GSM8K, Minerva MATH, IFEval, plus HumanEval/MBPP **and**
their EvalPlus Plus variants via lm-eval task names `humaneval_plus` /
`mbpp_plus` (datasets `evalplus/humanevalplus`, `evalplus/mbppplus`). Suites
`core` / `code` / `fastdllm` / `custom` select subsets. Decode optional pins
mirror Hub `threshold` / greedy / DualCache. Throughput records
`decode_profile` and refuses to imply fused-kernel parity in SUMMARY text.

`scripts/slurm/lm_eval.sbatch` now defaults to **4×4** GPUs. Each node runs
`accelerate launch` with `num_machines=NUM_NODES`, `num_processes=16` when
fully packed; only machine rank 0 writes throughput JSON and SUMMARY.

### 6.2 Prefetch honesty

On the login node we successfully cached and TaskManager-loaded:
`gsm8k`, `ifeval`, `humaneval`, `humaneval_plus`, `mbpp`, `mbpp_plus`.  
**GPQA** failed (Hub gated). **minerva_math** failed pending `lm-eval[math]`.
**MMLU** group load hit an incomplete `hails/mmlu_no_train` cache. Therefore
today’s C0 “roof” job is intentionally a **partial** paper suite, not a fake
full average.

### 6.3 Job genealogy (this day)

| Job ID | Name / role | Resources | Outcome |
|--------|-------------|-----------|---------|
| `1692667` | `lever_fastdllm_v2_masked` pre-fix | 4×4 | **Cancelled**; loss −6→−135; same sign bug |
| `1691076` | C2 `bqwen-eval` | 1 GPU | **Cancelled** (invalid ckpt science) |
| `1691158` | C2 `bqwen-lm-eval` | 1 GPU | **Cancelled** (loop spam) |
| `1694554` | `fastdllm_v2` retrain **after fix** | 4×4 | **Running**; early loss **+6.39**, later samples near **+2.4** (n≈3114 logged hits by evening) — opposite of C2’s negative trajectory |
| `1694594` | C0 lm-eval attempt | 1 node | **Failed** immediately: `Unknown SUITE=custom` before suite allow-list fix |
| `1694653` | C0 lm-eval 1×4 | 1×4 | **Cancelled** when moving default to 4 nodes |
| `1694730` | C0 lm-eval roof | **4×4** | **Running** (Accelerate multi-GPU; tasks `gsm8k,ifeval,humaneval,humaneval_plus,mbpp,mbpp_plus`); logs `slurm_logs/bqwen-lm-C0_1694730.{out,err}` |

**Invalidated artifacts (do not put in main tables):**  
`ar2block_masked_1660576` @ 6k fluency/lm-eval; any `1692667` checkpoints;
narrative claims “C2 > C0 recreate success.”

**Still valid as floor / control:**  
`ar2block_masked_1660579` train + ELBO + (pending) post-fix lm-eval.

### 6.4 Qualitative sample references (for the thesis appendix)

From `outputs/block_qwen/ar2block_masked_1660579/eval/samples.txt`, sample 0
opens roughly:

> `<|im_start|>ink>` then English CoT about restaurant options A–F …

Tails in the same file degrade into mixed scripts and replacement characters —
consistent with the bucket statistics above. These excerpts should be cited as
**free-gen diagnostics**, not as the primary conversion metric.

Live chat-condensed example (C0 greedy, afternoon probe): after a chat-template
`2+2` prompt the model emitted `4` then further `<|im_end|>` / think debris —
useful as a “competence with weak stopping” illustration.

---

## 7. Interpretation for the conversion thesis

The conversion story after today is:

1. **Hooks-off AR→block finetuning (C0) learns a nontrivial denoising model** on
   Nemotron chat packs (ELBO, masked-site accuracy, conditional continuity).
2. **Long bare free-gen is a severe OOD + format-prior test** and will look unfairly
   bad if used as the headline metric; think-tag fragmentation after
   `<|im_start|>` is a concrete mechanism, not mysticism.
3. **Fast-dLLM-shaped training on our stack was unsafe until the shift+plain_ce
   sign fix.** Post-fix `1694554` is the first honest exactness attempt;
   early positive losses are necessary but not sufficient for recreate success.
4. **Reporting must move to chat lm-eval (+ optional DualCache/confidence decode)**
   on multi-node Accelerate if we want commensurability with Fast-dLLM v2 tables,
   while remaining honest that tok/s is not fused-kernel parity.

---

## 8. Code and documentation touched (with intent)

| Path | Intent |
|------|--------|
| `src/discrete_diffusion/algorithms/block_trainer.py` | Make shift+plain_ce minimize-oriented |
| `tests/test_block_shift_loss.py` | Lock the sign contract in CI |
| `src/discrete_diffusion/evaluations/block_qwen_lm_eval.py` | Accelerate ranks; Hub-like threshold/greedy/DualCache pins; rank0 metrics write |
| `src/discrete_diffusion/evaluations/decode_throughput.py` | Named decode profiles + honesty note in JSON |
| `configs/eval/decode_throughput.yaml` | Profile defaults |
| `examples/block_qwen/lm_eval.sh` | Suites, Plus tasks, multi-node launch, custom TASKS |
| `scripts/slurm/lm_eval.sbatch` | Default 4×4 + srun launchers |
| `scripts/prefetch_lm_eval_data.sh` | Offline booster caches for paper-ish tasks |
| `experiments/2026-09-04_reference_recreation/EXPERIMENT.md` | Collapse section + pointer here |
| `experiments/README.md` | Index + day-log convention |
| `.cursor/rules/daily-experiment-logs.mdc` | Force thesis-depth daily logs going forward |

---

## 9. Open questions and next actions

1. Finish **1694730** and paste GSM8K / IFEval / HumanEval± / MBPP± numbers into
   an Updates section below — that is C0’s true conversion-floor scorecard.
2. Finish **1694554**; verify loss never goes negative; then lm-eval with
   `DECODE_PROFILE=dual_cache UNMASK_THRESHOLD=0.9 FORCE_GREEDY=1` on 4×4.
3. Close prefetch gaps: Hub login for GPQA; `pip install 'lm-eval[math]'`; repair
   MMLU cache for a fuller average column.
4. Consider a diagnostic free-gen variant that injects
   `<|im_start|>assistant\n` (or full chat) so appendix samples are not dominated
   by think-tag scrap modes.
5. Decide whether to keep gen-PPL only in BlockGen-native chapters going forward.

---

## 10. Thesis scrapbook (hedged prose, reusable)

**Failure analysis (engineering).**  
When a diffusion objective is expressed as an ELBO that internally uses a
negative cross-entropy factor, special-casing “plain CE” requires returning to a
**positive** CE (or equivalently applying an explicit sign flip). Reusing the
intermediate `−CE` tensor under a minimizer inverts learning. In our conversion
stack this bug appeared precisely on the Fast-dLLM-relevant setting
`shift_loss_targets ∧ plain_ce`, producing extreme logits and deterministic
string loops that are easy to misread as a conceptual failure of block
conversion.

**Evaluation methodology.**  
For instruction-tuned AR→block models trained on chat-formatted corpora,
unconditional long free-gen from a bare speech-act BOS token probes format
priors and long-horizon ancestral error accumulation more than it probes the
denoising objective optimized at train time. Conditional lm-evaluation with
chat templates is closer both to the training distribution and to published
Fast-dLLM reporting practice. Metrics that score only the first chunk of a long
free-gen canvas can systematically understate length-progressive degradation.

**Limitations.**  
Even after the sign fix, our DualCache / hierarchical KV path is an algorithmic
reimplementation; throughput numbers must not be presented as reproductions of
Fast-dLLM fused CUDA speedups. Partial lm-eval suites (missing GPQA/MMLU/MATH)
must not be silently averaged into a “paper score.” Pre-fix C2/fastdllm
checkpoints are excluded from main result tables.

**Positive program (pending completion of 1694554 / 1694730).**  
C0 remains the hooks-off conversion reference. Post-fix fastdllm_v2 is the
exactness cell for recipe fidelity. Native BlockGen OWT remains a separate
family under `LINE=block` and is not evidenced by this day’s conversion
forensics.

---

## 11. End-of-day status board

| Item | State at log write-up |
|------|------------------------|
| plain_ce sign fix + test | **Merged in working tree** |
| C2 / old fastdllm science | **Invalidated** |
| fastdllm_v2 retrain `1694554` | **Running**, loss positive (~6.4 → ~2.4 so far) |
| C0 lm-eval `1694730` | **Running** on 4×4, partial suite |
| C0 retrain needed? | **No** |
| Ready to claim recreate success? | **No** — wait for post-fix lm-eval |

### Updates (append as jobs finish)

**2026-09-07 morning postmortem** — see
[2026-09-07_lm_eval_nccl_timeout](../2026-09-07_lm_eval_nccl_timeout/EXPERIMENT.md).

- **`1694554`:** reached `max_steps=1900` with loss **+6.39 → ~1.1**, never
  negative; `last.ckpt` written; auto-eval **`1698982` COMPLETED** (gen-PPL
  ≈72; ELBO bs32 BPD ≈3.94). Slurm **FAILED** exit 6 was NCCL teardown after
  success — keep the run.
- **`1694730`:** GSM8K `generate_until` largely finished (~1328 Q&A prints;
  rank0 `tok_s≈2.04` over 4.65 h) then died on
  `wait_for_everyone` / NCCL **ALLREDUCE Timeout(ms)=600000**. No SUMMARY.
  Harness timeouts raised to 6 h in `block_qwen_lm_eval.py` + sbatch/sh.
