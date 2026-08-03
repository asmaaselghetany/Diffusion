# scripts/

SLURM job launchers (Kalyan-style `.sbatch` templates).

## Qwen block training — two pipelines

| Pipeline | Scripts | Submit |
|----------|---------|--------|
| **1. AR→block** (pretrained Qwen) | `slurm/ar2block_{masked,uniform}.sbatch` | `./scripts/submit_ar2block.sh both` |
| **2. Pure block** (scratch init) | `slurm/blockgen_{masked,uniform}.sbatch` | `./scripts/submit_blockgen.sh both` |

```bash
./scripts/submit_ar2block.sh both
./scripts/submit_blockgen.sh both
./scripts/resume_block_qwen.sh outputs/block_qwen/masked_<jobid>
./scripts/resume_block_qwen.sh outputs/block_qwen/blockgen_uniform_<jobid>
```

Shared: `_block_qwen_env.bash`, `_block_qwen_launch.bash`, `_resolve_block_qwen_run.bash`.

Legacy aliases: `slurm/masked.sbatch`, `slurm/uniform.sbatch` → Pipeline 1.

## Other

| Script | Purpose |
|--------|---------|
| `train_block_qwen_verify.sbatch` | GPU verification (S0–S6 via `smoke.sh`) |
| `train_block_qwen_masked.sbatch` | Alias → `slurm/masked.sbatch` |
| `train_block_qwen_uniform.sbatch` | Alias → `slurm/uniform.sbatch` |

Logs: `slurm_logs/%x_%j.out`
