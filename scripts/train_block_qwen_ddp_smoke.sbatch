#!/usr/bin/env bash
#SBATCH --job-name=bqwen-ddp-smoke
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new

set -euo pipefail

export ASMAA_WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash
cd "${REPO_ROOT}"
echo "=== DDP legacy probe (expect world_size=1 under SlurmEnvironment) ==="
srun --ntasks=1 --gpus-per-task=4 --cpu-bind=cores python -u tools/smoke_ddp.py
echo "=== 1-step block_masked train (external-rank launch — expect world_size=4) ==="
export NUM_NODES=1
export GPUS_PER_NODE=4
export NUM_GPUS=4
export LINE="${LINE:-ar2block}"
# Skip heavy last.ckpt on smoke (24G Lustre write desynced ranks / NCCL timeout).
# Do not use callbacks=null — Hydra rejects null for the callbacks config group.
export HYDRA_OVERRIDES="trainer.max_steps=1 trainer.num_sanity_val_steps=0 trainer.limit_val_batches=0 ~callbacks.save_last_on_train_end callbacks.checkpoint_every_n_steps.save_last=false callbacks.checkpoint_every_n_steps.every_n_train_steps=999999 callbacks.checkpoint_monitor.save_top_k=0 checkpointing.resume_from_ckpt=false"
export RUN_ROOT="${REPO_ROOT}/outputs/block_qwen/ddp_smoke_${SLURM_JOB_ID}"
export WANDB_MODE=disabled
export AUTO_RESUME=0
export RUN_FULL_EVAL=false
# Abandoned writers from timed-out smokes must not block for the 7-day default.
export DISCRETE_DIFFUSION_CACHE_STALE_SECONDS="${DISCRETE_DIFFUSION_CACHE_STALE_SECONDS:-900}"
# shellcheck disable=SC1091
source scripts/_block_qwen_launch.bash masked
