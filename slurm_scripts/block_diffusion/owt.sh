#!/bin/bash
#SBATCH --job-name=bd_owt
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --chdir=/e/project1/scifi/elsayed3/JEDi
#SBATCH --output=outputs/block_diffusion/logs/owt_%j.out
#SBATCH --error=outputs/block_diffusion/logs/owt_%j.err
#
# BlockDiffusion on OpenWebText (single full Booster node).
#
# Booster QOS (part_booster) caps wall time at 12h. Default MAX_STEPS=100k is
# enough for a solid OWT run on one job; BD3-LM paper uses 150k fine-tune
# (from MDLM init) or 1M total — override with MAX_STEPS= if needed.
#
# Submit:
#   sbatch slurm_scripts/block_diffusion/owt.sh
#   MODE=uniform sbatch slurm_scripts/block_diffusion/owt.sh
#   FROM_PRETRAINED=1 sbatch slurm_scripts/block_diffusion/owt.sh   # MDLM init (Baseline B)
#   FROM_AR_PRETRAINED=1 sbatch slurm_scripts/block_diffusion/owt.sh   # AR init
#   sbatch slurm_scripts/block_diffusion/owt_from_pretrain.sh
#   sbatch slurm_scripts/block_diffusion/owt_from_ar_pretrain.sh
#
# Multi-node (example: 2 nodes = 8 GPUs):
#   NUM_NODES=2 sbatch --nodes=2 slurm_scripts/block_diffusion/owt.sh

set -euo pipefail

# SBATCH --chdir sets cwd to repo root; BASH_SOURCE breaks under SLURM (spool copy).
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/block_diffusion"
# shellcheck source=jupiter_paths.sh
source "${SCRIPT_DIR}/jupiter_paths.sh"

export MODE="${MODE:-masked}"
export REPO_ROOT="${REPO_ROOT:-$(resolve_jedi_repo_root)}"
export OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/block_diffusion/owt}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_ROOT}/data_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-block_diffusion_owt}"

export NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE=4
export DATA="${DATA:-openwebtext-split}"
export SEQ_LEN="${SEQ_LEN:-1024}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-100}"
export CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-10000}"
# flex needs Triton — install via setup_cuda_venv.sh (PyTorch cu126 index, not plain pip).
export ATTN_BACKEND="${ATTN_BACKEND:-flex}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export RESAMPLE=1
export GENERATE_VAL_SAMPLES="${GENERATE_VAL_SAMPLES:-0}"

mkdir -p "${REPO_ROOT}/outputs/block_diffusion/logs" "${OUTPUT_BASE}/logs"

bash "${SCRIPT_DIR}/train_core.sh"
