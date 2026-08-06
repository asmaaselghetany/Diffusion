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
| Complementary per-block masks | `algo.complementary_masks` | `false` | Fast-dLLM v2; Li et al. 2025b (cited there) |
| Multi block-size training | `algo.block_size_mixture` | `[]` | BlockGen (arxiv [2606.02241](https://arxiv.org/abs/2606.02241)) |
| Stratified block-size / timestep draw | `algo.stratified_gamma` | `null` | BlockGen §3.2 |
| AR-informed predictor–corrector | `sampling.use_arpc` | `false` | BlockGen §3.3 (uniform decode) |

**Not implemented** (would require separate experiments, not neutral hooks): Fast-dLLM partial within-block masking, sub-block size 8, hierarchical KV cache; 64-GPU DeepSpeed.

### Enable a reference recipe (optional)

Do **not** put arm-specific hooks in `block_qwen.yaml` — Hydra dot-keys at experiment root (`algo.shift_loss_targets: true`) do not merge into `config.algo`. Pass overrides at launch:

```bash
# Fast-dLLM-style masked training hooks
HYDRA_OVERRIDES="algo.shift_loss_targets=true algo.complementary_masks=true" \
  sbatch scripts/slurm/masked.sbatch

# BlockGen-style uniform extras (fixed block 32 still; mixture adds more sizes)
HYDRA_OVERRIDES="algo.block_size_mixture=[16,32] algo.stratified_gamma=0.5 sampling.use_arpc=true" \
  sbatch scripts/slurm/uniform.sbatch
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
| Global batch | 256 (64× A100, ZeRO-3) | **128** (1 GPU × grad_accum) |
| Steps | ~6000 (paper) | **7500** (~2B tokens) |

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
| **2. blockgen** | Pure block diffusion (BlockGen style) | scratch (same Qwen arch) | `blockgen_{masked,uniform}.sbatch` | `blockgen_{masked,uniform}_<jobid>/` |

```bash
source env.sh && cd "$REPO_ROOT"

# Pipeline 1 — AR→block
./scripts/submit_ar2block.sh both

# Pipeline 2 — pure block diffusion
./scripts/submit_blockgen.sh both
```

Optional paper hooks (not for the neutral compare) can still be passed via `HYDRA_OVERRIDES` — see above.

### Resume after a 24h SLURM slot

A new `sbatch` gets a **new job id** and a **new output folder** unless you pin the run directory:

```bash
./scripts/resume_block_qwen.sh outputs/block_qwen/masked_126234
./scripts/resume_block_qwen.sh outputs/block_qwen/uniform_126242
./scripts/resume_block_qwen.sh outputs/block_qwen/blockgen_masked_<jobid>
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
sbatch scripts/slurm/eval_checkpoint.sbatch blockgen uniform <jobid>
sbatch scripts/slurm/eval_checkpoint.sbatch \
  outputs/block_qwen/masked_<jobid>/checkpoints/last.ckpt
```

**Not included:** Fast-dLLM task benches (GSM8K / MMLU / HumanEval) — different harness; not wired here yet.

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
