#!/bin/bash
#SBATCH --job-name=350_jepa_study_orchestrator
#SBATCH --partition=accelerated
#SBATCH --nodes=128
#SBATCH --gres=gpu:4
##SBATCH --reservation=llmtum
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/350_study_orchestrator_%j.out
#SBATCH --error=logs/350_study_orchestrator_%j.err

###############################################################################
# JEPA Diffusion Full Hyperparameter Study - Production Parallel Version
#
# This script orchestrates a complete hyperparameter study with:
# - Multi-node parallel execution with configurable GPUs per config
# - Automatic phase progression with checkpoint passing
# - Resume support: pass existing study dir to continue from where it stopped
# - Comprehensive logging and metrics extraction
# - Stage 2 decoder training with best Stage 1 checkpoint
#
# GPU Configuration:
#   GPUS_PER_CONFIG=1  → 64 parallel configs (fast exploration, single GPU)
#   GPUS_PER_CONFIG=2  → 32 parallel configs (medium batches)
#   GPUS_PER_CONFIG=4  → 16 parallel configs (full batch size, 4 GPUs each)
#
# Usage:
#   # New study (fresh start)
#   sbatch run_full_study_parallel.sh
#
#   # Resume existing study
#   RESUME_STUDY_DIR=/path/to/existing/study sbatch run_full_study_parallel.sh
#
#   # Custom configuration
#   GPUS_PER_CONFIG=2 WANDB_PROJECT=my_study sbatch run_full_study_parallel.sh
#
###############################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/sweep_outputs}"

# Model size configuration (180m or 350m)
MODEL_SIZE="${MODEL_SIZE:-350m}"

# Resume support: if RESUME_STUDY_DIR is set, continue that study
if [[ -n "${RESUME_STUDY_DIR:-}" ]]; then
    if [[ ! -d "${RESUME_STUDY_DIR}" ]]; then
        echo "ERROR: Resume directory does not exist: ${RESUME_STUDY_DIR}"
        exit 1
    fi
    STUDY_DIR="${RESUME_STUDY_DIR}"
    STUDY_NAME=$(basename "${STUDY_DIR}")
    RESUME_MODE=true
    echo "=========================================="
    echo "RESUME MODE: Continuing study ${STUDY_NAME}"
    echo "=========================================="
else
    STUDY_NAME="${STUDY_NAME:-jepa_${MODEL_SIZE}_full_study_$(date +%Y%m%d_%H%M%S)}"
    STUDY_DIR="${OUTPUT_BASE}/${STUDY_NAME}"
    RESUME_MODE=false
fi

cd "${REPO_ROOT}"
mkdir -p "${STUDY_DIR}/logs" logs

# ============================================================================
# Environment Setup
# ============================================================================
echo "=========================================="
echo "Setting up environment..."
echo "=========================================="

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
export WANDB_PROJECT="${WANDB_PROJECT:-jepa_${MODEL_SIZE}_hparam_study}"
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

# ============================================================================
# Multi-Node and GPU Configuration
# ============================================================================
NUM_NODES="${SLURM_NNODES:-128}"
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# GPUS_PER_CONFIG: How many GPUs each hyperparameter config uses
GPUS_PER_CONFIG="${GPUS_PER_CONFIG:-4}"

# Validate GPUS_PER_CONFIG
if [[ $((GPUS_PER_NODE % GPUS_PER_CONFIG)) -ne 0 ]]; then
    echo "ERROR: GPUS_PER_CONFIG (${GPUS_PER_CONFIG}) must evenly divide GPUS_PER_NODE (${GPUS_PER_NODE})"
    exit 1
fi

# Calculate derived values
CONFIGS_PER_NODE=$((GPUS_PER_NODE / GPUS_PER_CONFIG))
CONFIGS_PER_BATCH=$((TOTAL_GPUS / GPUS_PER_CONFIG))

echo "=========================================="
echo "JEPA Diffusion Hyperparameter Tuning Study"
echo "=========================================="
echo "Study: ${STUDY_NAME}"
echo "Model Size: ${MODEL_SIZE}"
echo "Output: ${STUDY_DIR}"
echo "Resume Mode: ${RESUME_MODE}"
echo ""
echo "Cluster Configuration:"
echo "  Nodes: ${NUM_NODES}"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo "  Total GPUs: ${TOTAL_GPUS}"
echo ""
echo "Hyperparameter Sweep Configuration:"
echo "  GPUs per config: ${GPUS_PER_CONFIG}"
echo "  Configs per node: ${CONFIGS_PER_NODE}"
echo "  Parallel configs per batch: ${CONFIGS_PER_BATCH}"
echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================================
# Helper: Check if a config has completed successfully
# ============================================================================
is_config_complete() {
    local phase_dir="$1"
    local config_idx="$2"
    local configs_file="${phase_dir}/configs.json"
    
    if [[ ! -f "${configs_file}" ]]; then
        return 1
    fi
    
    # Get run_name for this config
    local run_name=$(python -c "
import json
with open('${configs_file}') as f:
    configs = json.load(f)
for cfg in configs:
    if cfg['idx'] == ${config_idx}:
        print(cfg['run_name'])
        break
" 2>/dev/null)
    
    if [[ -z "${run_name}" ]]; then
        return 1
    fi
    
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
# Helper: Extract best run and checkpoint from phase
# ============================================================================
get_best_checkpoint() {
    local phase_dir="$1"
    local metric_name="${2:-val/nll}"
    local metric_goal="${3:-minimize}"
    
    python -c "
import json
from pathlib import Path

phase_dir = Path('${phase_dir}')
summary_file = phase_dir / 'summary.json'

if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)
    best_run_id = summary.get('best_run_id', '')
    if best_run_id:
        ckpt = phase_dir / 'runs' / best_run_id / 'checkpoints' / 'last.ckpt'
        if ckpt.exists():
            print(str(ckpt))
            exit(0)
        # Try best.ckpt as fallback
        ckpt = phase_dir / 'runs' / best_run_id / 'checkpoints' / 'best.ckpt'
        if ckpt.exists():
            print(str(ckpt))
            exit(0)

# Fallback: scan all runs
best_metric = float('inf') if '${metric_goal}' == 'minimize' else float('-inf')
best_ckpt = ''

runs_dir = phase_dir / 'runs'
if runs_dir.exists():
    for run_dir in runs_dir.iterdir():
        status_file = run_dir / 'status.json'
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
            if status.get('exit_code', 1) == 0:
                metrics = status.get('metrics', {})
                metric_val = metrics.get('${metric_name}')
                if metric_val is not None:
                    if '${metric_goal}' == 'minimize' and metric_val < best_metric:
                        best_metric = metric_val
                        ckpt = run_dir / 'checkpoints' / 'last.ckpt'
                        if ckpt.exists():
                            best_ckpt = str(ckpt)
                    elif '${metric_goal}' == 'maximize' and metric_val > best_metric:
                        best_metric = metric_val
                        ckpt = run_dir / 'checkpoints' / 'last.ckpt'
                        if ckpt.exists():
                            best_ckpt = str(ckpt)

print(best_ckpt)
" 2>/dev/null || echo ""
}

# ============================================================================
# Run Sweep Function (Multi-Node Parallel Execution)
# ============================================================================
run_sweep() {
    local sweep_config="$1"
    local num_runs="$2"
    local phase_dir="$3"
    local checkpoint="${4:-}"
    
    local config_path="${REPO_ROOT}/configs/sweep/${sweep_config}.yaml"
    
    if [[ ! -f "${config_path}" ]]; then
        echo "ERROR: Sweep config not found: ${config_path}"
        return 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Phase: ${sweep_config}"
    echo "Configurations: ${num_runs}"
    echo "Output: ${phase_dir}"
    echo "GPUs per config: ${GPUS_PER_CONFIG}"
    echo "Parallel configs: ${CONFIGS_PER_BATCH}"
    if [[ -n "${checkpoint}" ]]; then
        echo "Checkpoint: ${checkpoint}"
    fi
    echo "Started: $(date)"
    echo "=========================================="
    
    mkdir -p "${phase_dir}/runs" "${phase_dir}/logs"
    
    # Step 1: Generate configurations (skip if resuming and configs exist)
    local configs_file="${phase_dir}/configs.json"
    
    if [[ -f "${configs_file}" && "${RESUME_MODE}" == "true" ]]; then
        echo "Using existing configurations from ${configs_file}"
    else
    echo "Generating ${num_runs} configurations..."
    
    python -c "
import json
import sys
sys.path.insert(0, '${REPO_ROOT}/src')
from discrete_diffusion.tuning.sweep_config import load_sweep_config

config = load_sweep_config('${config_path}')
configs = config.generate_configs(num_runs=${num_runs}, seed=42)

# Add checkpoint if provided
checkpoint = '${checkpoint}'
if checkpoint and config.requires_checkpoint:
    key = config.requires_checkpoint.get('checkpoint_key', 'training.finetune_path')
    for cfg in configs:
        cfg[key] = checkpoint

# Save with run names
output = []
for idx, cfg in enumerate(configs):
    run_name = config.get_run_name(cfg, idx)
    output.append({
        'idx': idx,
        'run_name': run_name,
        'config': cfg
    })

with open('${phase_dir}/configs.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f'Generated {len(output)} configurations')
"
    fi
    
    local num_configs=$(python -c "import json; print(len(json.load(open('${phase_dir}/configs.json'))))")
    
    # Count completed and pending configs
    local completed=0
    local pending=0
    for idx in $(seq 0 $((num_configs - 1))); do
        if is_config_complete "${phase_dir}" "${idx}"; then
            ((completed+=1))
        else
            ((pending+=1))
        fi
    done
    
    echo "Configuration status: ${completed}/${num_configs} completed, ${pending} pending"
    
    if [[ ${pending} -eq 0 ]]; then
        echo "All configurations already completed. Skipping phase."
        return 0
    fi
    
    # Step 2: Run configurations in parallel across all nodes/GPUs
    echo ""
    echo "Launching ${pending} pending configurations (${GPUS_PER_CONFIG} GPU(s) each)..."
    echo "Using dynamic scheduling to maximize GPU utilization."
    echo ""
    
    # Track running jobs and their slots
    declare -A RUNNING_JOBS  # PID -> SLOT_ID
    declare -a SLOT_BUSY     # SLOT_ID -> busy (true/false)
    
    # Initialize slots
    for ((i=0; i<CONFIGS_PER_BATCH; i++)); do
        SLOT_BUSY[$i]=false
    done
    
    for config_idx in $(seq 0 $((num_configs - 1))); do
        # Skip completed configs
        if is_config_complete "${phase_dir}" "${config_idx}"; then
            continue
        fi
        
        # Find a free slot
        local free_slot=-1
        while [[ $free_slot -eq -1 ]]; do
            # Check for any free slots
            for ((i=0; i<CONFIGS_PER_BATCH; i++)); do
                if [[ "${SLOT_BUSY[$i]}" == "false" ]]; then
                    free_slot=$i
                    break
                fi
            done
            
            if [[ $free_slot -eq -1 ]]; then
                # No free slots, wait for at least one job to finish
                # Note: wait -n returns non-zero if a job fails, but we continue anyway
                # since we check PIDs to determine which jobs actually finished
                wait -n || true
                
                # Update SLOT_BUSY by checking which PIDs are still running
                for pid in "${!RUNNING_JOBS[@]}"; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        local finished_slot="${RUNNING_JOBS[$pid]}"
                        SLOT_BUSY[$finished_slot]=false
                        unset "RUNNING_JOBS[$pid]"
                    fi
                done
            fi
        done
        
        local slot=$free_slot
        SLOT_BUSY[$slot]=true
        
        # Calculate node and GPU range for this slot
        local node_idx=$((slot / CONFIGS_PER_NODE))
        local slot_on_node=$((slot % CONFIGS_PER_NODE))
        local first_gpu=$((slot_on_node * GPUS_PER_CONFIG))
        local target_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n "$((node_idx + 1))p")
        
        # Build GPU list for this config
        local gpu_list=""
        for g in $(seq 0 $((GPUS_PER_CONFIG - 1))); do
            if [[ -n "${gpu_list}" ]]; then
                gpu_list="${gpu_list},$((first_gpu + g))"
            else
                gpu_list="$((first_gpu + g))"
            fi
        done
        
        echo "[$(date +%T)] Launching config ${config_idx} in slot ${slot} on ${target_node} (GPUs: ${gpu_list})"
        
        # Launch config
        srun --nodes=1 --ntasks=1 \
            --nodelist=${target_node} \
            --output="${phase_dir}/logs/config_${config_idx}.out" \
            --error="${phase_dir}/logs/config_${config_idx}.err" \
            bash -c "
                # GPU visibility for this config
                export CUDA_VISIBLE_DEVICES=${gpu_list}
                
                # NCCL Configuration
                export NCCL_DEBUG=WARN
                export NCCL_IB_TIMEOUT=50
                export NCCL_IB_RETRY_CNT=10
                export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
                export NCCL_SOCKET_FAMILY=AF_INET
                export NCCL_SOCKET_IFNAME=lo
                export GLOO_SOCKET_IFNAME=lo
                export NCCL_IB_DISABLE=1
                
                # Cache directories
                export TORCH_HOME=${REPO_ROOT}/.cache/torch
                export TORCHINDUCTOR_CACHE_DIR=${REPO_ROOT}/.cache/torch_inductor
                export XDG_CACHE_HOME=${REPO_ROOT}/.cache
                
                # HuggingFace
                export HF_HOME=/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface
                export HF_DATASETS_CACHE=\${HF_HOME}/datasets
                export HF_HUB_OFFLINE=1
                
                # WandB
                export WANDB_MODE=${WANDB_MODE:-online}
                
                # Unset SLURM variables
                unset SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NAME
                unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_NNODES
                unset SLURM_LOCALID SLURM_PROCID SLURM_NODEID
                unset SLURM_JOB_NODELIST SLURM_NODELIST
                unset SLURM_STEP_GPUS SLURM_STEP_NUM_TASKS
                unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK
                
                # Python environment
                cd ${REPO_ROOT}
                source ${REPO_ROOT}/venv/bin/activate
                export PYTHONPATH=${REPO_ROOT}/src
                export HYDRA_FULL_ERROR=1
                
                python -m discrete_diffusion.tuning.run_single_config \
                    --configs-file '${phase_dir}/configs.json' \
                    --config-idx ${config_idx} \
                    --output-dir '${phase_dir}' \
                    --wandb-project '${WANDB_PROJECT}' \
                    --gpus-per-config ${GPUS_PER_CONFIG}
            " &
        
        RUNNING_JOBS[$!]=$slot
    done
    
    # Wait for all remaining jobs
    echo "Waiting for all remaining jobs to complete..."
    wait
    echo "All jobs in phase completed at $(date)"
    
    # Step 3: Aggregate results and find best run
    echo ""
    echo "Aggregating results..."
    python -c "
import json
from pathlib import Path

phase_dir = Path('${phase_dir}')
configs_file = phase_dir / 'configs.json'
with open(configs_file) as f:
    configs = json.load(f)

results = []
completed = 0
failed = 0
best_metric = float('inf')  # Assume minimization for val/nll
best_run_id = ''
best_run_metrics = {}

metric_name = 'val/nll'

for cfg_data in configs:
    run_name = cfg_data['run_name']
    run_dir = phase_dir / 'runs' / run_name
    status_file = run_dir / 'status.json'
    
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)
        results.append(status)
        
        if status.get('exit_code', 1) == 0:
            completed += 1
            # Extract metrics for best run selection
            metrics = status.get('metrics', {})
            metric_val = metrics.get(metric_name)
            if metric_val is not None and metric_val < best_metric:
                best_metric = metric_val
                best_run_id = run_name
                best_run_metrics = metrics
        else:
            failed += 1
    else:
        failed += 1
        results.append({'run_name': run_name, 'status': 'missing'})

summary = {
    'sweep_name': '${sweep_config}',
    'total_runs': len(configs),
    'completed': completed,
    'failed': failed,
    'metric_name': metric_name,
    'best_run_id': best_run_id,
    'best_run_metrics': best_run_metrics,
    'best_metric_value': best_metric if best_metric != float('inf') else None,
    'results': results
}

with open(phase_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f'Summary: {completed} completed, {failed} failed out of {len(configs)}')
if best_run_id:
    print(f'Best run: {best_run_id} with {metric_name}={best_metric:.4f}')
"
    
    echo ""
    echo "Phase ${sweep_config} completed"
    echo "Finished: $(date)"
    echo "=========================================="
    
    return 0
}

# ============================================================================
# Main Study Execution - Stage 1 Phases
# ============================================================================

# Phase 1A: Encoder Architecture (Primary exploration)
echo ""
echo "################################################################"
echo "# PHASE 1A: Encoder Architecture Sweep"
echo "################################################################"
run_sweep "stage1_${MODEL_SIZE}_encoder" 60 "${STUDY_DIR}/phase1a_encoder"
PHASE1A_STATUS=$?

if [[ ${PHASE1A_STATUS} -ne 0 ]]; then
    echo "WARNING: Phase 1A had errors (exit code: ${PHASE1A_STATUS})"
fi

# Get best checkpoint from Phase 1A for later Stage 2
PHASE1A_BEST_CKPT=$(get_best_checkpoint "${STUDY_DIR}/phase1a_encoder" "val/nll" "minimize")
echo "Phase 1A best checkpoint: ${PHASE1A_BEST_CKPT:-'(none found)'}"

# Phase 2A: Predictor Architecture (grid search, 72 max configs, constraints reduce to ~56)
echo ""
echo "################################################################"
echo "# PHASE 2A: Predictor Architecture Sweep"
echo "################################################################"
run_sweep "stage1_${MODEL_SIZE}_predictor" 72 "${STUDY_DIR}/phase2a_predictor"
PHASE2A_STATUS=$?

if [[ ${PHASE2A_STATUS} -ne 0 ]]; then
    echo "WARNING: Phase 2A had errors (exit code: ${PHASE2A_STATUS})"
fi

# Phase 2B: Training Hyperparameters
echo ""
echo "################################################################"
echo "# PHASE 2B: Training Hyperparameters Sweep"
echo "################################################################"
run_sweep "stage1_${MODEL_SIZE}_training" 40 "${STUDY_DIR}/phase2b_training"
PHASE2B_STATUS=$?

if [[ ${PHASE2B_STATUS} -ne 0 ]]; then
    echo "WARNING: Phase 2B had errors (exit code: ${PHASE2B_STATUS})"
fi

# Phase 2C: EMA Configuration (grid search, 54 total configs)
echo ""
echo "################################################################"
echo "# PHASE 2C: EMA Configuration Sweep"
echo "################################################################"
run_sweep "stage1_${MODEL_SIZE}_ema" 54 "${STUDY_DIR}/phase2c_ema"
PHASE2C_STATUS=$?

if [[ ${PHASE2C_STATUS} -ne 0 ]]; then
    echo "WARNING: Phase 2C had errors (exit code: ${PHASE2C_STATUS})"
fi

# ============================================================================
# Stage 2: Decoder Training with Best Stage 1 Checkpoint
# ============================================================================

# Find overall best Stage 1 checkpoint by comparing all phases
echo ""
echo "################################################################"
echo "# Selecting Best Stage 1 Checkpoint for Stage 2"
echo "################################################################"

BEST_STAGE1_CKPT=""
BEST_STAGE1_METRIC=99999

for phase_dir in "${STUDY_DIR}/phase1a_encoder" "${STUDY_DIR}/phase2a_predictor" "${STUDY_DIR}/phase2b_training" "${STUDY_DIR}/phase2c_ema"; do
    if [[ -f "${phase_dir}/summary.json" ]]; then
        metric=$(python -c "
import json
with open('${phase_dir}/summary.json') as f:
    s = json.load(f)
val = s.get('best_metric_value')
if val is not None:
    print(val)
else:
    print('999999')
" 2>/dev/null)
        ckpt=$(get_best_checkpoint "${phase_dir}" "val/nll" "minimize")
        if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
            if (( $(echo "${metric} < ${BEST_STAGE1_METRIC}" | bc -l) )); then
                BEST_STAGE1_METRIC="${metric}"
                BEST_STAGE1_CKPT="${ckpt}"
                echo "New best: ${ckpt} (val/nll=${metric})"
            fi
        fi
    fi
done

if [[ -n "${BEST_STAGE1_CKPT}" && -f "${BEST_STAGE1_CKPT}" ]]; then
    echo ""
    echo "Best Stage 1 checkpoint: ${BEST_STAGE1_CKPT}"
    echo "Best Stage 1 val/nll: ${BEST_STAGE1_METRIC}"
    
    # Phase 3: Readout Decoder Sweep (Stage 2)
    echo ""
    echo "################################################################"
    echo "# PHASE 3: Stage 2 Readout Decoder Sweep"
    echo "################################################################"
    run_sweep "stage2_${MODEL_SIZE}_readout" 60 "${STUDY_DIR}/phase3_decoder" "${BEST_STAGE1_CKPT}"
    PHASE3_STATUS=$?
    
    if [[ ${PHASE3_STATUS} -ne 0 ]]; then
        echo "WARNING: Phase 3 had errors (exit code: ${PHASE3_STATUS})"
    fi
else
    echo "WARNING: No valid Stage 1 checkpoint found. Skipping Stage 2."
    PHASE3_STATUS=1
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "################################################################"
echo "# STUDY COMPLETE"
echo "################################################################"
echo ""
echo "Study: ${STUDY_NAME}"
echo "Output: ${STUDY_DIR}"
echo ""
echo "Phase Status:"
echo "  Phase 1A (Encoder):    Exit ${PHASE1A_STATUS:-N/A}"
echo "  Phase 2A (Predictor):  Exit ${PHASE2A_STATUS:-N/A}"
echo "  Phase 2B (Training):   Exit ${PHASE2B_STATUS:-N/A}"
echo "  Phase 2C (EMA):        Exit ${PHASE2C_STATUS:-N/A}"
echo "  Phase 3 (Decoder):     Exit ${PHASE3_STATUS:-N/A}"
echo ""
echo "End time: $(date)"
echo "================================================================"

# Generate combined analysis
python -c "
import json
from pathlib import Path
from datetime import datetime

study_dir = Path('${STUDY_DIR}')
phases = [
    'phase1a_encoder', 
    'phase2a_predictor', 
    'phase2b_training', 
    'phase2c_ema',
    'phase3_decoder'
]

combined = {
    'study_name': '${STUDY_NAME}',
    'completed_at': datetime.now().isoformat(),
    'best_stage1_checkpoint': '${BEST_STAGE1_CKPT:-}',
    'best_stage1_metric': ${BEST_STAGE1_METRIC:-99999},
    'phases': {}
}

total_completed = 0
total_failed = 0

for phase in phases:
    summary_file = study_dir / phase / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            phase_summary = json.load(f)
            combined['phases'][phase] = phase_summary
            total_completed += phase_summary.get('completed', 0)
            total_failed += phase_summary.get('failed', 0)

combined['total_completed'] = total_completed
combined['total_failed'] = total_failed
combined['total_runs'] = total_completed + total_failed

with open(study_dir / 'study_summary.json', 'w') as f:
    json.dump(combined, f, indent=2, default=str)

print(f'Study summary saved to: {study_dir}/study_summary.json')
print(f'Total runs: {total_completed + total_failed} ({total_completed} completed, {total_failed} failed)')
" 2>/dev/null || echo "Could not generate combined summary"

echo "================================================================"
