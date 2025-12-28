#!/bin/bash
#SBATCH --job-name=jepa_diffusion_tuning
#SBATCH --partition=accelerated-h200,accelerated-h100,accelerated
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --reservation=llmtum
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=0-59%8
#SBATCH --output=logs/tuning_%A_%a.out
#SBATCH --error=logs/tuning_%A_%a.err

###############################################################################
# JEPA Diffusion Hyperparameter Tuning - Slurm Job Array
###############################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
SWEEP_DIR="${SWEEP_DIR:-${REPO_ROOT}/sweep_outputs/default}"
SWEEP_NAME="${SWEEP_NAME:-$(basename ${SWEEP_DIR})}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"

cd "${REPO_ROOT}"
mkdir -p logs

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
echo "JEPA Sweep - Array Task ${TASK_ID}"
echo "=========================================="
echo "Sweep: ${SWEEP_NAME}"
echo "Nodes: ${SLURM_NNODES}"
echo "Master: ${MASTER_NODE} (${MASTER_ADDR}:${MASTER_PORT})"
echo "Python: $(which python)"
echo "Start time: $(date)"
echo "=========================================="

DATA_CACHE="${REPO_ROOT}/data_cache"
mkdir -p "${DATA_CACHE}"
export DATA_CACHE MASTER_ADDR MASTER_PORT

# ============================================================================
# Run Single Configuration
# ============================================================================
srun --kill-on-bad-exit=1 --export=ALL bash -c 'torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=4 \
    --node_rank=${SLURM_NODEID} \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m discrete_diffusion.tuning.run_array_task \
    --configs-file "'"${SWEEP_DIR}"'/array_configs.json" \
    --task-id '"${TASK_ID}"' \
    --output-dir "'"${SWEEP_DIR}"'" \
    --sweep-name "'"${SWEEP_NAME}"'"'

echo "Task ${TASK_ID} completed at $(date)"
