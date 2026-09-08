# Reference recreation sanity — Fast-dLLM + BlockGen

**Date:** 2026-09-04 (updated 2026-09-06)  
**Repo:** `Diffusion-new` @ Jupiter booster  
**Goal:** Recreate published recipes with *our* components; if metrics land in the
pre-registered band, the pipeline is scientifically sane.

**2026-09-06 worklog:** [collapse / plain_ce fix / multi-node lm-eval](../2026-09-06_collapse_plain_ce_eval/EXPERIMENT.md).

**Family rule (locked):** Fast-dLLM = **conversion** (`LINE=ar2block`, masked).
BlockGen = **native** (`LINE=block`, scratch). Never cross-tag. Geometry on
conversion is `xfer_*` only (not claimed here).

## Match criteria (locked before looking)

| Cell | Family | Claim | Train recipe | Primary metric | Pass |
|------|--------|-------|--------------|----------------|------|
| **fastdllm_v2** | conversion | Fast-dLLM v2 **exactness** cell | AR→block + shift + complementary(**fused**) + `fast_dllm` + plain CE + **single_stream** hierarchical/DualCache + confidence decode; Nemotron; **1900 steps ≈ 1B toks** | lm-eval GSM8K + IFEval (+ HumanEval); samples not soup | Beats **C0**; Hub decode graph |
| **C2_fdllm_full** | conversion | Fast-dLLM-ish train-only control | same train levers **without** decode pins; 6k steps | lm-eval GSM8K + IFEval | Beats **C0** |
| **blockgen_owt_uniform** | native | BlockGen OWT 1+16 one-to-one | scratch **`line=block`** + uniform + `jdeschena/openwebtext` + weights/u-strat/pure_noise/CE@1 + ARPC; GBS 512 / len 1024 / lr 3e-4 | Val BPD trend + ARPC samples + gen-PPL | Trains cleanly; non-soup text; gen-PPL ≪ pre-fix soup (~266+) |
| **C0** (control) | conversion | Neutral conversion | hooks off | same lm-eval | Reference floor for C2 / fastdllm_v2 |
| **N0** (optional native floor) | native | Scratch uniform hooks-off | `line=block`, no BlockGen mixture | Val BPD + samples | Sanity vs BlockGen recipe cell |

**Not claimed:** bit-for-bit Block-DiT / GPT-2 vocab; fused flex train kernels (throughput).

### Fast-dLLM exactness protocol (sanity of our implementation)

**Judge train correctness** with Hub `modeling.py` levers (shift, complementary fused 2B, `fast_dllm`, plain CE, dual-stream concat, shared RoPE, block-align pack).  
**Judge decode correctness** only when `fdllm_confidence_decode` + `single_stream_decode` are on (thr=0.9, greedy, while-masks-remain, Hub `eval_block_diff_mask`).

| Axis | Official | Ours (`fastdllm_v2`) | Status |
|------|----------|----------------------|--------|
| Shift + complementary + plain CE + `fast_dllm` | Hub `modeling.py` | levers pinned | **exact intent** |
| Dual-stream `concat(xt,x0)` + shared RoPE | Hub train | `BlockTrainer` + Qwen block | **exact intent** |
| Complementary batching | fused `2B` cat | **fused** (`algo.complementary_batching=fused`; sequential=OOM fallback) | **closed** |
| Confidence decode | while-masks + thr + force-max | same | **exact intent** |
| Decode attn graph | single-stream block-causal + KV | `sampling.single_stream_decode` + DualCache | **closed** |
| DualCache | Hub `replace_position` | single-stream splice (same algorithm; not fused CUDA) | **closed (algo)** |
| Token budget | ~1B | **1900 × 256 × 2048 ≈ 1B** | matched |
| Data mix | LLaMA-Nemotron post-training (paper §4.1) | Nemotron SFT (`sft_qwen`) | **same family**; subset/packing may differ |
| Mask token | `\|<MASK>\|` id 151665 | `[MASK]` id 151665 | id match |

**Cancelled as inexact:** **1691161** (6k + no confidence decode); **1691206** (sequential complementary + dual-stream decode — superseded after fidelity close).

### BlockGen honest deltas (still “our components”)

| Axis | Official ([blockgen](https://github.com/jdeschena/blockgen)) | Ours |
|------|------|------|
| Data corpus + split | `jdeschena/openwebtext` `[:-100k]` / `[-100k:]` | **identical** via `data=openwebtext-blockgen` |
| Tokenizer | GPT-2 | Qwen2.5-1.5B-Instruct |
| Backbone | 170M Block-DiT scratch | Qwen-1.5B block scratch (`load_pretrained=false`) |
| Algo knobs | elbo + CE@1 + pure_noise[1] + 0.05/0.95 + u-stratified | **matched** (`blockgen_owt_uniform`) |
| Scale | GBS 512, len 1024, lr 3e-4, **1M** steps | same GBS/len/lr; 12h chunks + `AUTO_RESUME` toward 1M |
| Init line | scratch block | **`line=block` only** (registry refuses BlockGen presets on `ar2block`) |

**Discarded as BlockGen claim:** jobs **1660577** / **1660931** (used default
`LINE=ar2block` + Nemotron SFT — wrong family, wrong data). Those are at best
unlabeled **transfer**; do not put them in BlockGen tables. Eval jobs on
1660931 are diagnostic only.

## Jobs

| Role | Family | Submit | Job ID | Nodes | Run root |
|------|--------|--------|--------|-------|----------|
| DDP smoke (ops gate) | — | `train_block_qwen_ddp_smoke.sbatch` | **1660575** | 1×4 | `outputs/block_qwen/ddp_smoke_1660575` |
| **Fast-dLLM v2 exactness** | conversion | `./scripts/submit_fastdllm.sh` | **1694554** (post-`plain_ce` fix; ~1900) | 4×4 | `outputs/block_qwen/ar2block_masked_1694554` |
| **C2_fdllm_full (post-fix 6k)** | conversion | `C2_fdllm_full` | **1701099** | 4×4 | `outputs/block_qwen/ar2block_masked_1701099` |
| ~~C2_fdllm_full pre-fix~~ | conversion | `C2_fdllm_full` | ~~**1660576**~~ | 4×4 | invalidated (`plain_ce` sign; negative loss) |
| ~~fastdllm_v2 pre-fidelity~~ | — | cancelled (seq complementary / dual-stream decode) | ~~1691206~~ | — | — |
| ~~fastdllm_v2 inexact~~ | — | cancelled (no conf decode / 6k) | ~~1691161~~ | — | — |
| ~~BlockGen~~ (mis-wired, discard) | *was transfer-shaped* | `blockgen_uniform` + default ar2block | **1660577** / **1660931** | 4×4 | `…/ar2block_uniform_*` |
| Neutral C0 | conversion | `sbatch --nodes=4 … ar2block_masked.sbatch` | **1660579** | 4×4 | `outputs/block_qwen/ar2block_masked_1660579` |
| ar2block_uniform | conversion (B1) | main 4-arm | **1660932** | 4×4 | `outputs/block_qwen/ar2block_uniform_1660932` |
| block_masked | native (B2) | main 4-arm | **1660933** | 4×4 | `outputs/block_qwen/block_masked_1660933` |
| block_uniform | native (B2/N0-ish) | main 4-arm | **1660935** | 4×4 | `outputs/block_qwen/block_uniform_1660935` |
| **BlockGen OWT one-to-one** | native | `./scripts/submit_blockgen_owt.sh` | **1691059** | 4×4 | `outputs/block_qwen/block_uniform_1691059` |

Wall: 12h (booster QOS max). BlockGen OWT uses GBS 512 → longer wall; `AUTO_RESUME=1`.

### Prior failures (fixed before resubmit)

| Old job | Bug | Fix |
|---------|-----|-----|
| 1660387 | `callbacks=null` invalid Hydra group | disable save_last callbacks instead |
| 1660408 / 1660410 | GBS assert double-counted `num_nodes × WORLD_SIZE` | `_training_world_size` (codex) |
| 1660409 | `block_weights=0.05 0.0 …` split on spaces | Hydra list `[0.05,0.0,…]` in registry |
| 1660577 | `parse_block_weights` rejected OmegaConf `ListConfig` | accept ListConfig |
| 1660931 | BlockGen levers on `ar2block` + Nemotron | **family firewall** + `submit_blockgen_owt.sh` (`line=block`, OWT) |
| registry B3→ar2block | BlockGen extras locked to conversion | B3 / blockgen presets → `line=[block]`; `xfer_*` for transfer |

## Eval (after train)

Full suite = **samples + gen-PPL + ELBO** (`eval_checkpoint.sbatch`) **and** **lm-eval** (`lm_eval.sbatch`).
Default `SUITE=fastdllm` = Fast-dLLM v2 core (mmlu, gpqa, gsm8k, minerva_math, ifeval) **+** HumanEval(+)/MBPP(+) via lm-eval EvalPlus datasets. Throughput: `DECODE_PROFILE=baseline|hierarchical|dual_cache` (UNI-D2 DualCache port — **not** fused Fast-dLLM CUDA). Prefetch: `./scripts/prefetch_lm_eval_data.sh`.
**DepBench removed** from the pipeline (2026-09-06).

Jupiter eval/lm_eval sbatch fixed 2026-09-04 (booster, 12h wall). Submitted for finished runs:

| Run | Family note | eval job | lm_eval job |
|-----|-------------|----------|-------------|
| C0 1660579 | conversion | **1666335** (samples+genppl+elbo OK; failed on DepBench) | **1666338** (Hub offline) |
| Fast-dLLM 1660576 (resume **1690702** → 6k) | conversion C2 | **1691076** | **1691158** (gsm8k+ifeval+humaneval; Hub cached) |
| **fastdllm_v2 retrain (plain_ce fix)** | conversion | — | train **1694554** |
| **C0 lm-eval roof** | conversion floor | — | **1694730** (4×4 GPU Accelerate; gsm8k/ifeval/humaneval(+)/mbpp(+)) |
| 1660931 | **discard / not BlockGen** | **1666333** | **1666339** |
| ar2block_uniform 1660932 | conversion B1 | **1666331** | **1666341** |

Fast-dLLM train complete: **1690702** → `ar2block_masked_1660576/checkpoints/2-6000.ckpt`.

## Collapse investigation (2026-09-06)

**C0 (1660579) — train OK; sample issues are protocol + length, not C2-style collapse.**
- ELBO healthy (block-32 BPD ≈ 2.62). Hooks off. `trainer/loss` **+11.9 → ~+2** (correct sign).
- Offline free-gen (64×2048): latin **0.83 → 0.40** head→tail; tails pick byte junk (`�` ids 242/97). No residual MASK. gen-PPL ~146 is **`first_chunk_only`** (understates).
- **`ink>` mode (45/64):** pos0=`<|im_start|>`, pos1=`ink` (766). Nemotron pack never has BOS→`ink`; it *does* have `<think>` ≈ tokens `<th`(13708)+`ink`(766)+`>` (~26% of seqs). First-step logits after bare BOS rank-1 `<th` everywhere. Free-gen from inject_bos alone stitches think-tag scraps (`ink>`, `ink<think>`).
- **Short decode (≤192 toks):** rough but readable English CoT; not a hard loop. Nucleus 0.9 / greedy similar.
- **Conditional:** `"The sky is blue because"` → on-topic Rayleigh-ish text (greedy cleaner). Chat `2+2` greedy → answers **`4`** then overruns with `<|im_end|>` / digit spam (stop-token discipline weak).
- **Verdict:** keep C0 as conversion floor; do **not** retrain. Judge fluency on **chat-prefixed / short** gens, not 2048 free-gen tails. Primary C0 weakness = long free-gen error accumulation + think-tag prior under bare BOS, not objective sign bug.

**C2_fdllm_full (1660576 @ 6k) — hard mode collapse (root cause found).**
- lm-eval GSM8K: **49/49** = repeating `新人玩家` (token **117294**).
- **Not** a decode/shift-align bug: raw + aligned logits both rank-1 that token with logit **~779** (C0 ~15). Align off / double-align unchanged.
- Present by step **500** and worsens monotonically (logit 213→779; `lm_head` row norm 1.53→5.46). Top-k are Chinese game tokens.
- Token **absent** in first 20k Nemotron pack sequences — not a data-frequency mode.
- **Bug:** `_masked_loss` with `shift_loss_targets` + `plain_ce` returned `-ce`, while `nll`/`training_step` **minimize** without a further negate. Non-shift `plain_ce` correctly returns positive CE via `masked_plain_ce_per_token`. Result: C2 `trainer/loss` **-6 → -730** (maximizing CE → logit explosion).
- **Fix:** return `mask_positions * ce` for shift+plain_ce (2026-09-06). **Invalidate** C2 **1660576** / resume **1690702** and **fastdllm_v2 1692667** (cancelled; loss was already −6→−135). Retrain only after this patch.

**Before any re-run:** C0 needs no retrain. Next train = C2/fastdllm_v2 with fixed `plain_ce` sign. Prefer chat-template prefixes + early stop on `<|im_end|>` for lm-eval/sample probes.

```bash
# Conversion Fast-dLLM v2 (train + decode caches)
./scripts/submit_fastdllm.sh

# Legacy train-only ablation
./scripts/submit_lever.sh --preset C2_fdllm_full --arm masked --paper

# Native BlockGen OWT (only valid BlockGen recreate path)
./scripts/submit_blockgen_owt.sh

CKPT=.../checkpoints/last.ckpt
sbatch scripts/slurm/eval_checkpoint.sbatch "$CKPT"
sbatch --export=ALL,ASMAA_WORKSPACE,CKPT="$CKPT" scripts/slurm/lm_eval.sbatch
```
