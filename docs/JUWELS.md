# JUWELS / Jupiter setup (elsayed3)

Follows the same pattern as **JEDi** (`jupiter_paths.sh` + thin sbatch → launcher)
and **fourel1** (`scripts/slurm/_common.sh`).

## Paths

| What | Path |
|------|------|
| Repo (compute-visible) | `/e/project1/scifi/elsayed3/Diffusion` |
| Workspace env | `/e/project1/scifi/elsayed3/env.sh` |
| Scratch + HF cache | `/e/scratch/scifi/elsayed3/` |

**Never use `/p/project1` in Slurm jobs** — compute nodes only see `/e/`.

## One-time setup (login node)

```bash
source /e/project1/scifi/elsayed3/env.sh
cd /e/project1/scifi/elsayed3/Diffusion
bash scripts/setup_venv.sh
bash scripts/download_block_qwen_data.sh   # Qwen + Nemotron (login node, needs network)
```

After `git pull` from upstream (HAICORE paths): `bash scripts/setup_juwels.sh`

**DDP smoke** (after data download):

```bash
sbatch --account=scifi scripts/train_block_qwen_ddp_smoke.sbatch
```

## Submit

```bash
# Smoke (full node, 4 GPU)
sbatch --account=scifi scripts/train_block_qwen_verify.sbatch

# Paper arms (Pipeline 1 + 2)
./scripts/submit_ar2block.sh both
./scripts/submit_block.sh both

# Resume after 12h walltime
./scripts/resume_block_qwen.sh outputs/block_qwen/ar2block_masked_<jobid>
```

## Slurm layout

```
scripts/
  jupiter_paths.sh       # /e/ path helpers (like JEDi)
  slurm/_common.sh       # modules, venv, HF offline, DDP (like fourel1)
  _block_qwen_env.bash   # WandB + checkpoint checks
  _block_qwen_launch.bash # training launcher (4 GPU DDP)
  slurm/*.sbatch         # thin wrappers (--gres=gpu:4 full node)
```

## Booster conventions

- **Paper arms:** `--nodes=4 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=72` (16 GPUs)
- **Smoke / eval:** usually 1 node (`--ntasks=1 --gres=gpu:4`)
- **Account:** `--account=scifi` on every `sbatch` (including chained resume/eval)
- **Walltime:** `12:00:00` (booster QOS cap on Jupiter)
- **Modules:** `Stages/2026 GCC Python CUDA` (never pipe `module load` through sed)
- **Torch:** `torch==2.6.0+cu126` on aarch64 GH200 (see `setup_venv.sh`)
- **Grad accum:** global batch 256 → accum ≈ 16 with 16 GPUs (was 64 on 1 node)
