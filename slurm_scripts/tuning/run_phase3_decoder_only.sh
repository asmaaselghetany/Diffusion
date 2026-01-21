#!/bin/bash
#SBATCH --job-name=phase3_decoder_jepa_predictor
#SBATCH --partition=accelerated,accelerated-h200,accelerated-h100
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/phase3_decoder_jepa_predictor_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/phase3_decoder_jepa_predictor_%j.err

###############################################################################
# Phase 3 Decoder-Only Training Script
#
# This script runs ONLY Phase 3 (Stage 2 decoder training) using a specified
# Stage 1 checkpoint. Useful for:
# - Re-running decoder sweep with a known good Stage 1 checkpoint
# - Testing decoder configurations without running full study
#
# Usage:
#   # Edit STAGE1_CHECKPOINT below, then:
#   sbatch run_phase3_decoder_only.sh
#
#   # Or override via environment:
#   STAGE1_CHECKPOINT=/path/to/checkpoint.ckpt sbatch run_phase3_decoder_only.sh
###############################################################################

set -euo pipefail

# ============================================================================
# Configuration - EDIT THESE
# ============================================================================
REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
OUTPUT_BASE="${OUTPUT_BASE:-/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/sweep_outputs}"

# Model size configuration
MODEL_SIZE="${MODEL_SIZE:-180m}"

# **IMPORTANT**: Set your Stage 1 checkpoint path here
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/sweep_outputs/jepa_180m_full_study_20251223_202845/phase2a_predictor/runs/stage1_180m_predictor_0022_predictor_depth_3_predictor_hidden_size_512_predictor_n_heads_4_predictor_use_projections_False/checkpoints/best.ckpt}"

# Study naming
STUDY_NAME="${STUDY_NAME:-phase3_decoder_jepa_predictor_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)}"
STUDY_DIR="${OUTPUT_BASE}/${STUDY_NAME}"

# Resume support
if [[ -n "${RESUME_STUDY_DIR:-}" ]]; then
    STUDY_DIR="${RESUME_STUDY_DIR}"
    STUDY_NAME=$(basename "${STUDY_DIR}")
    RESUME_MODE=true
    echo "RESUME MODE: Continuing ${STUDY_NAME}"
else
    RESUME_MODE=false
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_BASE}" "${STUDY_DIR}/logs" logs

# ============================================================================
# Validate Checkpoint
# ============================================================================
if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "ERROR: Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}"
    exit 1
fi
echo "Using Stage 1 checkpoint: ${STAGE1_CHECKPOINT}"

# ============================================================================
# Environment Setup
# ============================================================================
module purge 2>/dev/null || true
module load devel/cuda/12.4 2>/dev/null || true

if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
    source "${REPO_ROOT}/venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${REPO_ROOT}/venv"
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src"
export WANDB_PROJECT="${WANDB_PROJECT:-jepa_best_predictor_${MODEL_SIZE}_phase3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1

# NCCL Configuration
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Cache directories
CACHE_DIR="${REPO_ROOT}/.cache"
mkdir -p "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch_inductor"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"

# ============================================================================
# GPU Configuration
# ============================================================================
NUM_NODES="${SLURM_NNODES:-8}"
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
GPUS_PER_CONFIG="${GPUS_PER_CONFIG:-4}"

if [[ $((GPUS_PER_NODE % GPUS_PER_CONFIG)) -ne 0 ]]; then
    echo "ERROR: GPUS_PER_CONFIG (${GPUS_PER_CONFIG}) must evenly divide GPUS_PER_NODE (${GPUS_PER_NODE})"
    exit 1
fi

CONFIGS_PER_NODE=$((GPUS_PER_NODE / GPUS_PER_CONFIG))
CONFIGS_PER_BATCH=$((TOTAL_GPUS / GPUS_PER_CONFIG))

echo "=========================================="
echo "Phase 3 Decoder Training Only"
echo "=========================================="
echo "Study: ${STUDY_NAME}"
echo "Output: ${STUDY_DIR}"
echo "Stage 1 Checkpoint: ${STAGE1_CHECKPOINT}"
echo ""
echo "Cluster: ${NUM_NODES} nodes × ${GPUS_PER_NODE} GPUs = ${TOTAL_GPUS} total"
echo "GPUs per config: ${GPUS_PER_CONFIG}, Parallel configs: ${CONFIGS_PER_BATCH}"
echo "Start: $(date)"
echo "=========================================="

# ============================================================================
# Helper: Check if config completed
# ============================================================================
is_config_complete() {
    local phase_dir="$1"
    local config_idx="$2"
    local configs_file="${phase_dir}/configs.json"
    
    [[ ! -f "${configs_file}" ]] && return 1
    
    local run_name=$(python -c "
import json
with open('${configs_file}') as f:
    configs = json.load(f)
for cfg in configs:
    if cfg['idx'] == ${config_idx}:
        print(cfg['run_name'])
        break
" 2>/dev/null)
    
    [[ -z "${run_name}" ]] && return 1
    
    local status_file="${phase_dir}/runs/${run_name}/status.json"
    if [[ -f "${status_file}" ]]; then
        local exit_code=$(python -c "
import json
with open('${status_file}') as f:
    print(json.load(f).get('exit_code', 1))
" 2>/dev/null)
        [[ "${exit_code}" == "0" ]] && return 0
    fi
    return 1
}

# ============================================================================
# Run Phase 3: Decoder Sweep
# ============================================================================
PHASE_DIR="${STUDY_DIR}/phase3_decoder"
SWEEP_CONFIG="stage2_${MODEL_SIZE}_readout"
NUM_RUNS=60
CONFIG_PATH="${REPO_ROOT}/configs/sweep/${SWEEP_CONFIG}.yaml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: Sweep config not found: ${CONFIG_PATH}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Phase 3: Decoder Training"
echo "Configurations: ${NUM_RUNS}"
echo "Output: ${PHASE_DIR}"
echo "Started: $(date)"
echo "=========================================="

mkdir -p "${PHASE_DIR}/runs" "${PHASE_DIR}/logs"

# Generate configurations
CONFIGS_FILE="${PHASE_DIR}/configs.json"

if [[ -f "${CONFIGS_FILE}" && "${RESUME_MODE}" == "true" ]]; then
    echo "Using existing configurations from ${CONFIGS_FILE}"
else
    echo "Generating ${NUM_RUNS} configurations..."
    python -c "
import json
import sys
sys.path.insert(0, '${REPO_ROOT}/src')
from discrete_diffusion.tuning.sweep_config import load_sweep_config

config = load_sweep_config('${CONFIG_PATH}')
configs = config.generate_configs(num_runs=${NUM_RUNS}, seed=42)

# Add checkpoint
checkpoint = '${STAGE1_CHECKPOINT}'
if checkpoint and config.requires_checkpoint:
    key = config.requires_checkpoint.get('checkpoint_key', 'training.finetune_path')
    for cfg in configs:
        cfg[key] = checkpoint

output = []
for idx, cfg in enumerate(configs):
    run_name = config.get_run_name(cfg, idx)
    output.append({'idx': idx, 'run_name': run_name, 'config': cfg})

with open('${CONFIGS_FILE}', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f'Generated {len(output)} configurations')
"
fi

NUM_CONFIGS=$(python -c "import json; print(len(json.load(open('${CONFIGS_FILE}'))))")

# Count completed/pending
completed=0
pending=0
for idx in $(seq 0 $((NUM_CONFIGS - 1))); do
    if is_config_complete "${PHASE_DIR}" "${idx}"; then
        ((completed+=1))
    else
        ((pending+=1))
    fi
done

echo "Status: ${completed}/${NUM_CONFIGS} completed, ${pending} pending"

if [[ ${pending} -eq 0 ]]; then
    echo "All configurations completed. Nothing to do."
    exit 0
fi

echo ""
echo "Launching ${pending} pending configurations..."

# Track running jobs
declare -A RUNNING_JOBS
declare -a SLOT_BUSY

for ((i=0; i<CONFIGS_PER_BATCH; i++)); do
    SLOT_BUSY[$i]=false
done

for config_idx in $(seq 0 $((NUM_CONFIGS - 1))); do
    is_config_complete "${PHASE_DIR}" "${config_idx}" && continue
    
    # Find free slot
    free_slot=-1
    while [[ $free_slot -eq -1 ]]; do
        for ((i=0; i<CONFIGS_PER_BATCH; i++)); do
            if [[ "${SLOT_BUSY[$i]}" == "false" ]]; then
                free_slot=$i
                break
            fi
        done
        
        if [[ $free_slot -eq -1 ]]; then
            wait -n || true
            for pid in "${!RUNNING_JOBS[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    SLOT_BUSY[${RUNNING_JOBS[$pid]}]=false
                    unset "RUNNING_JOBS[$pid]"
                fi
            done
        fi
    done
    
    slot=$free_slot
    SLOT_BUSY[$slot]=true
    
    node_idx=$((slot / CONFIGS_PER_NODE))
    slot_on_node=$((slot % CONFIGS_PER_NODE))
    first_gpu=$((slot_on_node * GPUS_PER_CONFIG))
    target_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n "$((node_idx + 1))p")
    
    gpu_list=""
    for g in $(seq 0 $((GPUS_PER_CONFIG - 1))); do
        [[ -n "${gpu_list}" ]] && gpu_list="${gpu_list},"
        gpu_list="${gpu_list}$((first_gpu + g))"
    done
    
    echo "[$(date +%T)] Config ${config_idx} → slot ${slot} on ${target_node} (GPUs: ${gpu_list})"
    
    srun --nodes=1 --ntasks=1 \
        --nodelist=${target_node} \
        --output="${PHASE_DIR}/logs/config_${config_idx}.out" \
        --error="${PHASE_DIR}/logs/config_${config_idx}.err" \
        bash -c "
            export CUDA_VISIBLE_DEVICES=${gpu_list}
            export NCCL_DEBUG=WARN
            export NCCL_IB_TIMEOUT=50
            export NCCL_IB_RETRY_CNT=10
            export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
            unset NCCL_SOCKET_IFNAME NCCL_SOCKET_FAMILY
            export NCCL_SOCKET_IFNAME=\"=lo\"
            export GLOO_SOCKET_IFNAME=\"lo\"
            export NCCL_IB_DISABLE=1
            export TORCH_HOME=${REPO_ROOT}/.cache/torch
            export TORCHINDUCTOR_CACHE_DIR=${REPO_ROOT}/.cache/torch_inductor
            export XDG_CACHE_HOME=${REPO_ROOT}/.cache
            export HF_HOME=/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface
            export HF_DATASETS_CACHE=\${HF_HOME}/datasets
            export HF_HUB_OFFLINE=1
            export WANDB_MODE=${WANDB_MODE:-online}
            unset SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NAME
            unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_NNODES
            unset SLURM_LOCALID SLURM_PROCID SLURM_NODEID
            unset SLURM_JOB_NODELIST SLURM_NODELIST
            unset SLURM_STEP_GPUS SLURM_STEP_NUM_TASKS
            cd ${REPO_ROOT}
            source ${REPO_ROOT}/venv/bin/activate
            export PYTHONPATH=${REPO_ROOT}/src
            export HYDRA_FULL_ERROR=1
            python -m discrete_diffusion.tuning.run_single_config \
                --configs-file '${PHASE_DIR}/configs.json' \
                --config-idx ${config_idx} \
                --output-dir '${PHASE_DIR}' \
                --wandb-project '${WANDB_PROJECT}' \
                --gpus-per-config ${GPUS_PER_CONFIG} \
                --num-nodes 1
        " &
    
    RUNNING_JOBS[$!]=$slot
done

echo "Waiting for all jobs..."
wait
echo "All jobs completed at $(date)"

# ============================================================================
# Aggregate Results
# ============================================================================
echo ""
echo "Aggregating results..."
python -c "
import json
from pathlib import Path

phase_dir = Path('${PHASE_DIR}')
with open(phase_dir / 'configs.json') as f:
    configs = json.load(f)

results = []
completed = 0
failed = 0
best_metric = float('inf')
best_run_id = ''
best_run_metrics = {}

for cfg_data in configs:
    run_name = cfg_data['run_name']
    status_file = phase_dir / 'runs' / run_name / 'status.json'
    
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)
        results.append(status)
        
        if status.get('exit_code', 1) == 0:
            completed += 1
            metrics = status.get('metrics', {})
            val_nll = metrics.get('val/nll')
            if val_nll is not None and val_nll < best_metric:
                best_metric = val_nll
                best_run_id = run_name
                best_run_metrics = metrics
        else:
            failed += 1
    else:
        failed += 1
        results.append({'run_name': run_name, 'status': 'missing'})

summary = {
    'sweep_name': 'phase3_decoder',
    'stage1_checkpoint': '${STAGE1_CHECKPOINT}',
    'total_runs': len(configs),
    'completed': completed,
    'failed': failed,
    'metric_name': 'val/nll',
    'best_run_id': best_run_id,
    'best_run_metrics': best_run_metrics,
    'best_metric_value': best_metric if best_metric != float('inf') else None,
    'results': results
}

with open(phase_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f'Summary: {completed} completed, {failed} failed out of {len(configs)}')
if best_run_id:
    print(f'Best run: {best_run_id} with val/nll={best_metric:.4f}')
"

echo ""
echo "=========================================="
echo "Phase 3 Decoder Training Complete"
echo "Output: ${PHASE_DIR}"
echo "End: $(date)"
echo "=========================================="