#!/usr/bin/env bash
#SBATCH --job-name=ts_jepa_train
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err

set -euo pipefail

export MODEL=latent_jepa
export REPO_ROOT="${REPO_ROOT:-/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi}"
export OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/tiny_shakespeare/baselines}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/datasets/}"
export WANDB_PROJECT="${WANDB_PROJECT:-tiny_shakespeare_baselines}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
export SEQ_LEN="${SEQ_LEN:-128}"

CORE_DIR="${REPO_ROOT}/slurm_scripts/shakespeare_baselines"
"${CORE_DIR}/train_core.sh"
