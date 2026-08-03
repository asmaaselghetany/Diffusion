# Block Qwen SLURM

Two training pipelines (hooks off on both; only init + corruption differ).

| Pipeline | Intent | Init | Scripts |
|----------|--------|------|---------|
| **1. ar2block** | AR→block (Fast-dLLM style) | pretrained Qwen Instruct | `ar2block_masked.sbatch`, `ar2block_uniform.sbatch` |
| **2. blockgen** | Pure block diffusion (BlockGen style) | scratch (same arch) | `blockgen_masked.sbatch`, `blockgen_uniform.sbatch` |

Legacy aliases `masked.sbatch` / `uniform.sbatch` → Pipeline 1 (for resume of `masked_*` / `uniform_*` runs).

## Resources

- 120G RAM, 24h, **2 GPUs** each (DDP)
- Data: Alpaca SFT (`sft_qwen`)
- Seq / block: 2048 / 32
- Steps: 7500

See [docs/BLOCK_QWEN_TRAINING.md](../../docs/BLOCK_QWEN_TRAINING.md).

### Submit

```bash
# Pipeline 1 — AR→block
./scripts/submit_ar2block.sh both

# Pipeline 2 — pure block diffusion
./scripts/submit_blockgen.sh both
```

### Resume (after 24h limit)

```bash
./scripts/resume_block_qwen.sh outputs/block_qwen/masked_126234
./scripts/resume_block_qwen.sh outputs/block_qwen/ar2block_uniform_<jobid>
./scripts/resume_block_qwen.sh outputs/block_qwen/blockgen_masked_<jobid>
```

## Eval / metrics

**During training** (every `val_check_interval`): `val/bpd` (ELBO), sample entropy, WandB sample table, saved `validation_samples/`.

**After a clean finish**: launch prepares max-step `last.ckpt` and auto-submits eval only if `global_step >= max_steps` → samples → gen-PPL → DepBench → ELBO under `<run>/eval/`. Disable with `RUN_FULL_EVAL=false`.

**Manual** (timeouts / re-runs):

```bash
sbatch scripts/slurm/eval_checkpoint.sbatch outputs/block_qwen/masked_126234/checkpoints/last.ckpt
sbatch scripts/slurm/eval_checkpoint.sbatch ar2block masked <jobid>
./scripts/submit_block_qwen_eval.sh runs
```

Fast-dLLM-style **task** scores (GSM8K / MMLU / HumanEval) are not in this suite yet.

## Outputs

| Pipeline | Directory |
|----------|-----------|
| Pipeline 1 (legacy) | `outputs/block_qwen/{masked,uniform}_<jobid>/` |
| Pipeline 1 | `outputs/block_qwen/ar2block_{masked,uniform}_<jobid>/` |
| Pipeline 2 | `outputs/block_qwen/blockgen_{masked,uniform}_<jobid>/` |

Logs: `slurm_logs/bqwen-*.out`
