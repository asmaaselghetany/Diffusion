#!/usr/bin/env bash
#SBATCH --job-name=ts_cont_train
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err

set -euo pipefail

export MODEL=continuous
export REPO_ROOT="${REPO_ROOT:-/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi}"
export OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/tiny_shakespeare/baselines}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/datasets/}"
export WANDB_PROJECT="${WANDB_PROJECT:-tiny_shakespeare_baselines}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-16}"
export SEQ_LEN="${SEQ_LEN:-128}"

export CED_HIDDEN_SIZE="${CED_HIDDEN_SIZE:-512}"
export CED_N_HEADS="${CED_N_HEADS:-8}"
export CED_N_BLOCKS="${CED_N_BLOCKS:-6}"
export CED_DECODER_N_BLOCKS="${CED_DECODER_N_BLOCKS:-2}"
export CED_ENCODER_NAME="${CED_ENCODER_NAME:-Qwen/Qwen3-Embedding-0.6B}"

CORE_DIR="${REPO_ROOT}/slurm_scripts/shakespeare_baselines"
"${CORE_DIR}/train_core.sh"
