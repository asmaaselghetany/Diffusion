#!/bin/bash
#SBATCH --job-name=jepa_sweep
#SBATCH --partition=accelerated-h200,accelerated-h100,accelerated
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --reservation=llmtum
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

###############################################################################
# Single Sweep Execution Script
# 
# Usage:
#   sbatch --export=SWEEP_CONFIG=stage1_180m_encoder,NUM_RUNS=60 run_single_sweep.sh
###############################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
SWEEP_CONFIG="${SWEEP_CONFIG:-stage1_180m_encoder}"
NUM_RUNS="${NUM_RUNS:-50}"
CHECKPOINT="${CHECKPOINT:-}"
SEED="${SEED:-42}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/sweep_outputs}"
SWEEP_DIR="${OUTPUT_BASE}/${SWEEP_CONFIG}_$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
mkdir -p "${SWEEP_DIR}/logs" logs

# ============================================================================
# Environment Setup
# ============================================================================
module purge 2>/dev/null || true
module load devel/cuda/12.4 2>/dev/null || true

# Activate virtual environment
source "${REPO_ROOT}/venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src"

# WandB configuration
export WANDB_PROJECT="${WANDB_PROJECT:-jepa_hparam_sweep}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# HuggingFace cache
export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# DEPRECATED: export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1

export HYDRA_FULL_ERROR=1

# ============================================================================
# NCCL and Network Configuration
# ============================================================================
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_FAMILY=AF_INET

RDZV_IFNAME=ib0
if ! ip link show dev "${RDZV_IFNAME}" >/dev/null 2>&1; then
    RDZV_IFNAME=eth0
    if ! ip link show dev "${RDZV_IFNAME}" >/dev/null 2>&1; then
        RDZV_IFNAME=$(ip -o link show | awk -F': ' '{print $2}' | grep -v lo | head -1)
    fi
fi
export GLOO_SOCKET_IFNAME="${RDZV_IFNAME}"
export NCCL_SOCKET_IFNAME="${RDZV_IFNAME}"

CACHE_DIR="${REPO_ROOT}/.cache"
mkdir -p "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch_inductor"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"

# ============================================================================
# Master Node Setup
# ============================================================================
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname -I | awk '{print $1}')
export MASTER_PORT=29500

echo "=========================================="
echo "JEPA Diffusion Hyperparameter Sweep"
echo "=========================================="
echo "Sweep: ${SWEEP_CONFIG}"
echo "Runs: ${NUM_RUNS}"
echo "Output: ${SWEEP_DIR}"
echo "Nodes: ${SLURM_NNODES}, GPUs per node: 4"
echo "Master: ${MASTER_NODE} (${MASTER_ADDR}:${MASTER_PORT})"
echo "Python: $(which python)"
echo "Start time: $(date)"
echo "=========================================="

DATA_CACHE="${REPO_ROOT}/data_cache"
mkdir -p "${DATA_CACHE}"
export DATA_CACHE MASTER_ADDR MASTER_PORT

# ============================================================================
# Run Sweep
# ============================================================================
CHECKPOINT_ARG=""
if [[ -n "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARG="--checkpoint ${CHECKPOINT}"
fi

python -m discrete_diffusion.tuning.sweep_runner \
    "${REPO_ROOT}/configs/sweep/${SWEEP_CONFIG}.yaml" \
    --output-dir "${SWEEP_DIR}" \
    --num-runs "${NUM_RUNS}" \
    --seed "${SEED}" \
    --wandb-project "${WANDB_PROJECT}" \
    ${CHECKPOINT_ARG}

echo ""
echo "=========================================="
echo "Sweep Completed: ${SWEEP_CONFIG}"
echo "Output: ${SWEEP_DIR}"
echo "End time: $(date)"
echo "=========================================="
