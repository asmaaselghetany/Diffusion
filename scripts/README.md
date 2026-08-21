# scripts/

Cluster launchers for the Qwen **four-arm** bakeoff (on top of UNI-D²).

## Four arms

| Pipeline | Arms | Submit |
|----------|------|--------|
| **AR→block** (pretrained Qwen) | `ar2block_masked`, `ar2block_uniform` | `./scripts/submit_ar2block.sh both` |
| **Scratch block** | `block_masked`, `block_uniform` | `./scripts/submit_block.sh both` |

```bash
./scripts/submit_ar2block.sh both
./scripts/submit_block.sh both
./scripts/resume_block_qwen.sh outputs/block_qwen/<run_dir>
```

Shared helpers: `_block_qwen_env.bash`, `_block_qwen_launch.bash`, `_block_qwen_ckpt.bash`, `_block_qwen_eval.bash`, `_resolve_block_qwen_run.bash`.

Sbatch templates live under `scripts/slurm/` (`ar2block_*.sbatch`, `block_*.sbatch`).  
Legacy `blockgen_*` / `submit_blockgen.sh` still resolve old run dirs only.

## Verify / eval

| Script | Purpose |
|--------|---------|
| `train_block_qwen_verify.sbatch` | GPU verification (`examples/block_qwen/smoke.sh`) |
| `submit_block_qwen_eval.sh` | Post-train eval wrapper |
| `slurm/arm_sanity.sbatch`, `pipeline_sanity.sbatch`, `copy_x0_probe.sbatch` | Probes |

Docs: [docs/BLOCK_PATH.md](../docs/BLOCK_PATH.md).
