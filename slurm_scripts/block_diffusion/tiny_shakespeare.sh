#!/bin/bash
#SBATCH --job-name=bd_tiny_shk
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --chdir=/e/project1/scifi/elsayed3/JEDi
#SBATCH --output=outputs/block_diffusion/logs/tiny_shakespeare_%j.out
#SBATCH --error=outputs/block_diffusion/logs/tiny_shakespeare_%j.err
#
# Quick BlockDiffusion run on Tiny Shakespeare (full Booster node, 4 GPUs).
#
# Submit:
#   sbatch slurm_scripts/block_diffusion/tiny_shakespeare.sh
#   MODE=uniform sbatch slurm_scripts/block_diffusion/tiny_shakespeare.sh
#
# Or dry-run the command without submitting:
#   DRY_RUN=1 bash slurm_scripts/block_diffusion/tiny_shakespeare.sh

set -euo pipefail

# SBATCH --chdir sets cwd to repo root; BASH_SOURCE breaks under SLURM (spool copy).
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/block_diffusion"
# shellcheck source=jupiter_paths.sh
source "${SCRIPT_DIR}/jupiter_paths.sh"

export MODE="${MODE:-masked}"
export REPO_ROOT="${REPO_ROOT:-$(resolve_jedi_repo_root)}"
export OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/block_diffusion/tiny_shakespeare}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_ROOT}/data_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-block_diffusion_tiny_shakespeare}"

export NUM_NODES=1
export GPUS_PER_NODE=4
export DATA=tiny_shakespeare
export SEQ_LEN="${SEQ_LEN:-128}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
export CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-500}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export GENERATE_VAL_SAMPLES="${GENERATE_VAL_SAMPLES:-0}"

mkdir -p "${REPO_ROOT}/outputs/block_diffusion/logs" "${OUTPUT_BASE}/logs"

bash "${SCRIPT_DIR}/train_core.sh"
