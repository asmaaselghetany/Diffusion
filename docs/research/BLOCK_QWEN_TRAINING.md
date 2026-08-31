# Block Qwen training

Block diffusion fine-tune of **Qwen2.5-Instruct** on instruction SFT data.

**Experiment config:** `block_qwen` — switch arm with `algo=block_masked` or `algo=block_uniform`.

## Neutral baseline (masked vs uniform)

The `block_qwen` experiment is a **controlled comparison**: both arms share the same model, data, sequence length, block size, optimizer, steps, batch, noise schedule, and sampler defaults. The **only** intentional difference is the forward corruption type (`masked` absorbing vs `uniform` over vocabulary).

All optional training / sampling hooks are **off** in both `configs/algo/block_*.yaml`.
Lever promotion is tracked in [`LEVERS.md`](LEVERS.md) — do not enable a paper hook
on the neutral recipe without a row + go/no-go there.

| Hook | Config key | Neutral value | Used in literature |
|------|------------|---------------|-------------------|
| Token shift at loss | `algo.shift_loss_targets` | `false` | Fast-dLLM v2 (arxiv [2509.26328](https://arxiv.org/abs/2509.26328)); also in this repo’s MDLM path |
| Complementary paired masks | `algo.complementary_masks` | `false` | Fast-dLLM v2 — **paired `m`/`~m`** via two sequential forwards (not polarity flip; not 2B batch) |
| Fast-dLLM mask schedule | `algo.mask_schedule` | `alpha` | Fast-dLLM: set `fast_dllm` for `p=(1-ε)t+ε` |
| Multi block-size (list) | `algo.block_size_mixture` | `[]` | BlockGen analogue (uniform list draw) |
| Weighted `2^k` mixture | `algo.block_weights` | `null` | BlockGen (XOR with list mixture) |
| Block-size per GPU | `algo.block_size_per_gpu` | `null` | BlockGen `same\|random\|u-stratified` |
| Stratified timestep draw | `algo.stratified_gamma` | `null` | Ours (≠ BlockGen u-stratified) |
| Pure noise at sizes | `algo.pure_noise_block_sizes` | `[]` | BlockGen |
| Per-size loss override | `algo.loss_type_special_cases` | `[]` | BlockGen e.g. `[1,ce]` |
| AR-informed PC | `sampling.use_arpc` / `arpc_mode` | `false` / `simplified` | BlockGen §3.3 (`simplified`\|`blockgen` + corruption modes; uniform; needs size 1) |
| Sub-block decode window | `sampling.sub_block_size` | `null` | Fast-dLLM `small_block_size` analogue |
| Hierarchical / progressive KV | `sampling.hierarchical_kv` | `false` | Truncated dual-stream (+ causal KV for ARPC) |
| DualCache splice | `sampling.use_block_cache` | `false` | Requires `hierarchical_kv`; K/V-only approx |
| Joint AR (C5) | `algo.joint_ar_alpha` | `0` | `L_AR + α L_diff`; epoch `val/nll` stays diffusion-only — use `val/joint_nll` |
| Causal clean AR | `algo.causal_clean_stream` | `false` | Requires `joint_ar_alpha>0`; true causal NLD-style stream |
| Hybrid p_uniform (B4) | `algo.hybrid_p_uniform` | `0.1` | Hybrid arm only; `hybrid_p10` pins default |

**Optional fidelity (default off):** BlockGen full ARPC (`arpc_mode=blockgen`); hierarchical truncate (`hierarchical_kv`); DualCache (`use_block_cache` via lever `dual_cache`).

### Enable a reference recipe (optional)

Do **not** put arm-specific hooks in `block_qwen.yaml` — Hydra dot-keys at experiment root (`algo.shift_loss_targets: true`) do not merge into `config.algo`. Prefer the **lever registry**:

```bash
# Fast-dLLM-style (corrected complementary)
./scripts/submit_lever.sh --preset fdllm --arm masked

# BlockGen-style uniform extras
./scripts/submit_lever.sh --preset blockgen_uniform --arm uniform

# List / dry-run
./scripts/submit_lever.sh --list
./scripts/submit_lever.sh --preset fdllm_shift_only --arm masked --dry-run
```

Legacy: pass overrides at launch only if you know what you are doing — registry validation will not run:

```bash
HYDRA_OVERRIDES="algo.shift_loss_targets=true algo.complementary_masks=true" \
  sbatch scripts/slurm/ar2block_masked.sbatch
```

Verify in the SLURM log that flags appear **inside** the `algo:` struct, not as orphaned root keys.

## Hyperparameters (shared)

| Setting | Literature anchor | `block_qwen` |
|---------|-------------------|--------------|
| Base model | Qwen2.5-1.5B-Instruct | same |
| Data | LLaMA-Nemotron post-training SFT (Fast-dLLM) | **same hub** (`nvidia/Llama-Nemotron-Post-Training-Dataset`, default splits `chat,safety,science`; Alpaca via `sft_qwen_alpaca`) |
| Seq length | 2048 | 2048 |
| Block size | 32 | 32 |
| LR | 2×10⁻⁵ | same |
| Warmup | 500 steps | same |
| Global batch | 256 (64× A100, ZeRO-3) | **256** (grad_accum on local GPUs) |
| Steps | ~6000 (paper) | **6000** (~3.15B tokens @ 256×2048) |

Fast-dLLM reports shift + complementary masks on top of this recipe; BlockGen reports mixture + stratified γ + ARPC for its uniform instantiation. Those are **not** part of the neutral `block_qwen` comparison.

### Data (Nemotron default)

`block_qwen` uses `configs/data/sft_qwen.yaml` → `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT), default splits **chat + safety + science**.

```bash
# optional: also pull capped code/math
export NEMOTRON_SFT_SPLITS=chat,safety,science,code,math
export NEMOTRON_SFT_MAX_PER_SPLIT=code=100000,math=100000

# rollback to Alpaca proxy
#   pass Hydra: data=sft_qwen_alpaca
```

First job downloads + tokenizes into `${scratch_dir}/block_qwen_sft_nemotron` (can take a while / substantial disk).

## Submit

Two pipelines (hooks off). Within each pipeline only corruption differs (masked vs uniform). Between pipelines only **init** differs.

| Pipeline | Intent | Init | Scripts | Outputs |
|----------|--------|------|---------|---------|
| **1. ar2block** | AR→block (Fast-dLLM style) | pretrained Qwen Instruct | `ar2block_{masked,uniform}.sbatch` | `ar2block_{masked,uniform}_<jobid>/` |
| **2. block** | Pure / scratch block diffusion | scratch (same Qwen arch) | `block_{masked,uniform}.sbatch` | `block_{masked,uniform}_<jobid>/` |

```bash
source env.sh && cd "$REPO_ROOT"

# Pipeline 1 — AR→block
./scripts/submit_ar2block.sh both

# Pipeline 2 — pure block diffusion
./scripts/submit_block.sh both
```

Optional paper hooks (not for the neutral compare) can still be passed via `HYDRA_OVERRIDES` — see above.

### Resume after a 24h SLURM slot

A new `sbatch` gets a **new job id** and a **new output folder** unless you pin the run directory:

```bash
./scripts/resume_block_qwen.sh outputs/block_qwen/masked_126234
./scripts/resume_block_qwen.sh outputs/block_qwen/uniform_126242
./scripts/resume_block_qwen.sh outputs/block_qwen/block_masked_<jobid>
```

This sets `RUN_ROOT` to the existing folder and loads `checkpoints/last.ckpt`.

### Smoke (~200 steps, seq 512)

```bash
RESUME_FROM_CKPT=false \
HYDRA_OVERRIDES="trainer.max_steps=200 model.length=512 block_size=16 loader.global_batch_size=8" \
sbatch scripts/slurm/ar2block_masked.sbatch
```

### Collapse early-stop

Off by default in `block_qwen` (paper-like train loop; no in-train samples). Enable with `eval.collapse_early_stop=true` **and** `eval.generate_samples=true` if you want sample-based early stop. Breadcrumb: `<run>/collapse_early_stop.json`.

## Metrics

### During training (`block_qwen` defaults)

| Metric | Source |
|--------|--------|
| `train/*`, `val/nll`, `val/bpd`, `val/ppl` | ELBO on train + cheap val (`limit_val_batches: 64`, every 500 opt steps) |
| `val/sample_entropy` + WandB sample table | `eval.generate_samples=true` |
| `validation_samples/step_*.pt` | `eval.save_validation_samples=true` |

### After training (auto)

When a job **exits 0**, [`_block_qwen_launch.bash`](../scripts/_block_qwen_launch.bash) runs `_block_qwen_prepare_last_ckpt` (highest `global_step` among valid ckpts → `last.ckpt`) and submits `eval_checkpoint.sbatch` **only if** that step is ≥ `trainer.max_steps`. Raw/stale `last.ckpt` is never trusted (Lightning often writes `last-v1` instead). Suite under `<run>/eval/`:

1. Samples (64)
2. Gen-PPL (`gpt2-large`)
3. DepBench
4. ELBO sweep over block sizes `{1,4,16,32}`

TIME LIMIT / crash / prepared step &lt; `max_steps` → no auto-eval (`./scripts/resume_block_qwen.sh <run_dir>`). Disable auto-eval with `RUN_FULL_EVAL=false`.

```bash
sbatch scripts/slurm/eval_checkpoint.sbatch masked <jobid>
sbatch scripts/slurm/eval_checkpoint.sbatch ar2block masked <jobid>
sbatch scripts/slurm/eval_checkpoint.sbatch block uniform <jobid>
sbatch scripts/slurm/eval_checkpoint.sbatch \
  outputs/block_qwen/masked_<jobid>/checkpoints/last.ckpt
```

**Task benches + tok/s (wired):** `examples/block_qwen/lm_eval.sh`
→ accuracy on MMLU / GPQA / GSM8K / Minerva Math / IFEval / HumanEval, plus
`tok_s.json` (dedicated) and `tok_s_lm_eval.json` (during generation).
See `<run>/lm_eval/SUMMARY.md`.

**Not included:** EvalPlus HumanEval+/MBPP+ (optional separate CLI later).

## Local (no SLURM)

```bash
source env.sh && source .venv/bin/activate
python -m discrete_diffusion +experiment=block_qwen algo=block_masked \
  checkpointing.save_dir=outputs/block_qwen_smoke_local \
  checkpointing.resume_from_ckpt=false \
  trainer.max_steps=200 model.length=512 block_size=16
```

## Literature map (corruption vs hooks)

| Work | Corruption | Training extras in paper | Decode extras |
|------|------------|--------------------------|---------------|
| **MDLM** (Sahoo et al., NeurIPS 2024, [2406.07524](https://arxiv.org/abs/2406.07524)) | Full-sequence masked / SUBS ELBO | Standard SUBS; no block hooks | Semi-AR sampler |
| **BlockGen** ([2606.02241](https://arxiv.org/abs/2606.02241)) | Masked **or** uniform **within block** | Block-size mixture γ, stratified multi-GPU draws | ARPC (uniform) |
| **Fast-dLLM v2** ([2509.26328](https://arxiv.org/abs/2509.26328)) | Block masked (partial mask in paper) | Shifted-label loss, complementary masks, concat(xt,x0) | Sub-block parallel decode, hierarchical cache |

Our neutral baseline isolates the BlockGen-style question — *masked vs uniform corruption within fixed block size* — without any row’s “extras” column enabled.
