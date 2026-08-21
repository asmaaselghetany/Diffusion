# Block Qwen SLURM

Two training pipelines (hooks off on both; only init + corruption differ).

| Pipeline | Intent | Init | Scripts |
|----------|--------|------|---------|
| **1. ar2block** | AR→block (Fast-dLLM style) | pretrained Qwen Instruct | `ar2block_masked.sbatch`, `ar2block_uniform.sbatch` |
| **2. block** | Scratch block diffusion | scratch (same arch) | `block_masked.sbatch`, `block_uniform.sbatch` |

Legacy: `blockgen_*.sbatch` / `submit_blockgen.sh` only for old `blockgen_*` run dirs.  
`masked.sbatch` / `uniform.sbatch` → Pipeline 1 (legacy names).

## Resources

- 120G RAM, 24h, **2 GPUs** each (DDP)
- Seq / block: 2048 / 32
- Steps: 6000

See [docs/research/BLOCK_QWEN_TRAINING.md](../../docs/research/BLOCK_QWEN_TRAINING.md).

### Submit

```bash
# Pipeline 1 — AR→block
./scripts/submit_ar2block.sh both

# Pipeline 2 — scratch block
./scripts/submit_block.sh both
```

### Resume (after 24h limit)

```bash
./scripts/resume_block_qwen.sh outputs/block_qwen/ar2block_uniform_<jobid>
./scripts/resume_block_qwen.sh outputs/block_qwen/block_masked_<jobid>
```

## Eval / metrics

Prefer portable eval: `bash examples/block_qwen/eval.sh <ckpt>` (upstream modules).

**Manual cluster:**

```bash
sbatch scripts/slurm/eval_checkpoint.sbatch outputs/block_qwen/block_masked_<jobid>/checkpoints/last.ckpt
sbatch scripts/slurm/eval_checkpoint.sbatch ar2block masked <jobid>
./scripts/submit_block_qwen_eval.sh runs
```

## Outputs

| Pipeline | Directory |
|----------|-----------|
| Pipeline 1 | `outputs/block_qwen/ar2block_{masked,uniform}_<jobid>/` |
| Pipeline 2 | `outputs/block_qwen/block_{masked,uniform}_<jobid>/` |
| Legacy P2 | `outputs/block_qwen/blockgen_{masked,uniform}_<jobid>/` |

Logs: `slurm_logs/bqwen-*.out`
