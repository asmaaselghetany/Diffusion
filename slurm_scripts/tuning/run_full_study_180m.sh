#!/bin/bash
#SBATCH --job-name=jepa_180m_tuning
#SBATCH --partition=accelerated-h200,accelerated-h100,accelerated
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
##SBATCH --reservation=llmtum
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/tuning_%j.out
#SBATCH --error=logs/tuning_%j.err

###############################################################################
# JEPA Diffusion Full Hyperparameter Study - 180M Model
# 
# This is a simplified version using the sweep_runner.py module.
# For production multi-node execution with more control, use:
#   run_full_study_parallel.sh
#
# Configuration: 8 nodes × 4 GPUs = 32 GPUs
# Each phase runs configs across available GPUs.
#
# Usage:
#   sbatch run_full_study_180m.sh                    # New study
#   RESUME_DIR=/path/to/study sbatch run_full_study_180m.sh  # Resume
###############################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/sweep_outputs}"

# Resume support
if [[ -n "${RESUME_DIR:-}" ]]; then
    STUDY_DIR="${RESUME_DIR}"
    STUDY_NAME=$(basename "${STUDY_DIR}")
    echo "Resuming study: ${STUDY_NAME}"
else
STUDY_NAME="jepa_180m_full_study_$(date +%Y%m%d_%H%M%S)"
STUDY_DIR="${OUTPUT_BASE}/${STUDY_NAME}"
fi

cd "${REPO_ROOT}"
mkdir -p "${STUDY_DIR}/logs" logs

# ============================================================================
# Environment Setup
# ============================================================================
module purge 2>/dev/null || true
module load devel/cuda/12.4 2>/dev/null || true

# Activate virtual environment
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
source "${REPO_ROOT}/venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${REPO_ROOT}/venv"
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src"

# WandB configuration
export WANDB_PROJECT="${WANDB_PROJECT:-jepa_180m_hparam_study}"
export WANDB_MODE="${WANDB_MODE:-online}"

# HuggingFace cache (offline mode for compute nodes)
export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1

# Hydra debug mode
export HYDRA_FULL_ERROR=1

# ============================================================================
# NCCL and Network Configuration
# ============================================================================
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_FAMILY=AF_INET

# Network interface selection
RDZV_IFNAME=ib0
if ! ip link show dev "${RDZV_IFNAME}" >/dev/null 2>&1; then
    RDZV_IFNAME=eth0
    if ! ip link show dev "${RDZV_IFNAME}" >/dev/null 2>&1; then
        RDZV_IFNAME=$(ip -o link show | awk -F': ' '{print $2}' | grep -v lo | head -1)
    fi
fi
export GLOO_SOCKET_IFNAME="${RDZV_IFNAME}"
export NCCL_SOCKET_IFNAME="${RDZV_IFNAME}"

# Cache directories
CACHE_DIR="${REPO_ROOT}/.cache"
mkdir -p "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch_inductor"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"

echo "=========================================="
echo "JEPA Diffusion Hyperparameter Tuning Study"
echo "=========================================="
echo "Study: ${STUDY_NAME}"
echo "Output: ${STUDY_DIR}"
echo "Nodes: ${SLURM_NNODES:-8}, GPUs per node: 4"
echo "Python: $(which python)"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================================
# Helper: Extract best checkpoint from phase
# ============================================================================
get_best_checkpoint() {
    local phase_dir="$1"
    
    python -c "
import json
from pathlib import Path

phase_dir = Path('${phase_dir}')
summary_file = phase_dir / 'summary.json'

best_ckpt = ''
if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)
    best_run_id = summary.get('best_run_id', '')
    if best_run_id:
        for name in ['last.ckpt', 'best.ckpt']:
            ckpt = phase_dir / 'runs' / best_run_id / 'checkpoints' / name
            if ckpt.exists():
                best_ckpt = str(ckpt)
                break

print(best_ckpt)
" 2>/dev/null || echo ""
}

# ============================================================================
# Run Sweep Function
# ============================================================================
run_sweep() {
    local sweep_config="$1"
    local num_runs="$2"
    local phase_dir="$3"
    local checkpoint="${4:-}"
    
    echo ""
    echo "=========================================="
    echo "Running sweep: ${sweep_config} (${num_runs} configs)"
    echo "Phase dir: ${phase_dir}"
    echo "Started: $(date)"
    if [[ -n "${checkpoint}" ]]; then
        echo "Checkpoint: ${checkpoint}"
    fi
    echo "=========================================="
    
    local ckpt_arg=""
    if [[ -n "${checkpoint}" ]]; then
        ckpt_arg="--checkpoint ${checkpoint}"
    fi
    
    python -m discrete_diffusion.tuning.sweep_runner \
        "${REPO_ROOT}/configs/sweep/${sweep_config}.yaml" \
        --output-dir "${phase_dir}" \
        --num-runs "${num_runs}" \
        --wandb-project "${WANDB_PROJECT}" \
        --parallel \
        ${ckpt_arg}
    
    return $?
}

# ============================================================================
# Phase Execution
# ============================================================================

# Phase 1A: Encoder Architecture
echo ""
echo "################################################################"
echo "# Phase 1A: Encoder Architecture Sweep"
echo "################################################################"
run_sweep "stage1_180m_encoder" 60 "${STUDY_DIR}/phase1a_encoder"
PHASE1A_STATUS=$?

# Get best checkpoint
PHASE1A_CKPT=$(get_best_checkpoint "${STUDY_DIR}/phase1a_encoder")

# Phase 2A: Predictor Architecture (grid, ~56 configs after constraints)
echo ""
echo "################################################################"
echo "# Phase 2A: Predictor Architecture Sweep"
echo "################################################################"
run_sweep "stage1_180m_predictor" 72 "${STUDY_DIR}/phase2a_predictor"
PHASE2A_STATUS=$?

PHASE2A_CKPT=$(get_best_checkpoint "${STUDY_DIR}/phase2a_predictor")

# Phase 2B: Training Hyperparameters
echo ""
echo "################################################################"
echo "# Phase 2B: Training Hyperparameters Sweep"
echo "################################################################"
run_sweep "stage1_180m_training" 40 "${STUDY_DIR}/phase2b_training"
PHASE2B_STATUS=$?

PHASE2B_CKPT=$(get_best_checkpoint "${STUDY_DIR}/phase2b_training")

# Phase 2C: EMA Configuration (grid, 54 configs)
echo ""
echo "################################################################"
echo "# Phase 2C: EMA Configuration Sweep"
echo "################################################################"
run_sweep "stage1_180m_ema" 54 "${STUDY_DIR}/phase2c_ema"
PHASE2C_STATUS=$?

PHASE2C_CKPT=$(get_best_checkpoint "${STUDY_DIR}/phase2c_ema")

# ============================================================================
# Select Best Stage 1 Checkpoint for Stage 2
# ============================================================================
echo ""
echo "################################################################"
echo "# Selecting Best Stage 1 Checkpoint"
echo "################################################################"

BEST_CKPT=""
BEST_METRIC=99999

for phase_dir in "${STUDY_DIR}/phase1a_encoder" "${STUDY_DIR}/phase2a_predictor" "${STUDY_DIR}/phase2b_training" "${STUDY_DIR}/phase2c_ema"; do
    if [[ -f "${phase_dir}/summary.json" ]]; then
        metric=$(python -c "
import json
with open('${phase_dir}/summary.json') as f:
    s = json.load(f)
val = s.get('best_value')
if val is not None:
    print(val)
else:
    print('999999')
" 2>/dev/null || echo "999999")
        
        ckpt=$(get_best_checkpoint "${phase_dir}")
        if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
            if (( $(echo "${metric} < ${BEST_METRIC}" | bc -l 2>/dev/null || echo 0) )); then
                BEST_METRIC="${metric}"
                BEST_CKPT="${ckpt}"
                echo "New best: ${ckpt} (metric=${metric})"
            fi
        fi
    fi
done

# ============================================================================
# Phase 3: Stage 2 Decoder Training
# ============================================================================
PHASE3_STATUS=0

if [[ -n "${BEST_CKPT}" && -f "${BEST_CKPT}" ]]; then
    echo ""
    echo "################################################################"
    echo "# Phase 3: Stage 2 Readout Decoder Sweep"
    echo "################################################################"
    echo "Using checkpoint: ${BEST_CKPT}"
    
    run_sweep "stage2_180m_readout" 60 "${STUDY_DIR}/phase3_decoder" "${BEST_CKPT}"
    PHASE3_STATUS=$?
else
    echo "WARNING: No valid Stage 1 checkpoint found. Skipping Stage 2."
    PHASE3_STATUS=1
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "Study Completed: ${STUDY_NAME}"
echo "Output: ${STUDY_DIR}"
echo ""
echo "Phase Status:"
echo "  Phase 1A (Encoder):    Exit ${PHASE1A_STATUS}"
echo "  Phase 2A (Predictor):  Exit ${PHASE2A_STATUS}"
echo "  Phase 2B (Training):   Exit ${PHASE2B_STATUS}"
echo "  Phase 2C (EMA):        Exit ${PHASE2C_STATUS}"
echo "  Phase 3 (Decoder):     Exit ${PHASE3_STATUS}"
echo ""
echo "Best Stage 1 Checkpoint: ${BEST_CKPT:-'(none)'}"
echo "End time: $(date)"
echo "=========================================="

# Generate combined summary
python -c "
import json
from pathlib import Path
from datetime import datetime

study_dir = Path('${STUDY_DIR}')
phases = ['phase1a_encoder', 'phase2a_predictor', 'phase2b_training', 'phase2c_ema', 'phase3_decoder']

combined = {
    'study_name': '${STUDY_NAME}',
    'completed_at': datetime.now().isoformat(),
    'best_stage1_checkpoint': '${BEST_CKPT:-}',
    'phases': {}
}

for phase in phases:
    summary_file = study_dir / phase / 'summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            combined['phases'][phase] = json.load(f)

with open(study_dir / 'study_summary.json', 'w') as f:
    json.dump(combined, f, indent=2, default=str)

print(f'Summary saved to {study_dir}/study_summary.json')
" 2>/dev/null || echo "Could not generate combined summary"
