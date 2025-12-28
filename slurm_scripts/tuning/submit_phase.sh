#!/bin/bash
###############################################################################
# Helper Script: Submit a Single Phase of the Hyperparameter Study
#
# Usage:
#   ./submit_phase.sh encoder         # Run encoder sweep
#   ./submit_phase.sh predictor       # Run predictor sweep
#   ./submit_phase.sh training        # Run training hparams sweep
#   ./submit_phase.sh ema             # Run EMA sweep
#   ./submit_phase.sh readout CKPT    # Run readout sweep with Stage 1 checkpoint
#   ./submit_phase.sh optimizer       # Run optimizer sweep
#   ./submit_phase.sh all             # Run all phases sequentially
#
###############################################################################

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
cd "${REPO_ROOT}"

# Ensure venv exists
if [[ ! -f "${REPO_ROOT}/venv/bin/activate" ]]; then
    echo "Warning: Virtual environment not found at ${REPO_ROOT}/venv"
    echo "Please create it first: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
fi

PHASE="${1:-encoder}"
CHECKPOINT="${2:-}"

# Create logs directory
mkdir -p logs

case "${PHASE}" in
    encoder)
        echo "Submitting encoder architecture sweep (60 runs)..."
        sbatch --export=SWEEP_CONFIG=stage1_180m_encoder,NUM_RUNS=60 \
            slurm_scripts/tuning/run_single_sweep.sh
        ;;
    
    predictor)
        echo "Submitting predictor architecture sweep (50 runs)..."
        sbatch --export=SWEEP_CONFIG=stage1_180m_predictor,NUM_RUNS=50 \
            slurm_scripts/tuning/run_single_sweep.sh
        ;;
    
    training)
        echo "Submitting training hyperparameters sweep (40 runs)..."
        sbatch --export=SWEEP_CONFIG=stage1_180m_training,NUM_RUNS=40 \
            slurm_scripts/tuning/run_single_sweep.sh
        ;;
    
    ema)
        echo "Submitting EMA configuration sweep (27 runs)..."
        sbatch --export=SWEEP_CONFIG=stage1_180m_ema,NUM_RUNS=27 \
            slurm_scripts/tuning/run_single_sweep.sh
        ;;
    
    readout)
        if [[ -z "${CHECKPOINT}" ]]; then
            echo "Error: readout phase requires Stage 1 checkpoint path"
            echo "Usage: ./submit_phase.sh readout /path/to/stage1/checkpoint"
            exit 1
        fi
        echo "Submitting readout decoder sweep (60 runs) with checkpoint: ${CHECKPOINT}"
        sbatch --export=SWEEP_CONFIG=stage2_180m_readout,NUM_RUNS=60,CHECKPOINT="${CHECKPOINT}" \
            slurm_scripts/tuning/run_single_sweep.sh
        ;;
    
    optimizer)
        echo "Submitting optimizer sweep (72 runs)..."
        sbatch --export=SWEEP_CONFIG=optimizer,NUM_RUNS=72 \
            slurm_scripts/tuning/run_single_sweep.sh
        ;;
    
    350m_encoder)
        echo "Submitting 350M encoder architecture sweep (50 runs)..."
        sbatch --export=SWEEP_CONFIG=stage1_350m_encoder,NUM_RUNS=50 \
            slurm_scripts/tuning/run_single_sweep.sh
        ;;
    
    all)
        echo "Submitting full 180M hyperparameter study..."
        sbatch slurm_scripts/tuning/run_full_study_180m.sh
        ;;
    
    generate)
        # Generate Slurm scripts without submitting
        SWEEP_CONFIG="${2:-stage1_180m_encoder}"
        NUM_RUNS="${3:-50}"
        OUTPUT_DIR="${4:-sweep_outputs/${SWEEP_CONFIG}}"
        
        echo "Generating Slurm array script for ${SWEEP_CONFIG}..."
        
        # Activate environment for local Python execution
        if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
            source "${REPO_ROOT}/venv/bin/activate"
        fi
        export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
        
        python -m discrete_diffusion.tuning.sweep_runner \
            "configs/sweep/${SWEEP_CONFIG}.yaml" \
            --output-dir "${OUTPUT_DIR}" \
            --num-runs "${NUM_RUNS}" \
            --generate-slurm
        
        echo ""
        echo "Generated scripts in ${OUTPUT_DIR}/"
        echo "Submit with: sbatch ${OUTPUT_DIR}/run_sweep.sh"
        ;;
    
    *)
        echo "Unknown phase: ${PHASE}"
        echo ""
        echo "Available phases:"
        echo "  encoder      - Stage 1 encoder architecture sweep"
        echo "  predictor    - Stage 1 predictor architecture sweep"
        echo "  training     - Stage 1 training hyperparameters sweep"
        echo "  ema          - Stage 1 EMA configuration sweep"
        echo "  readout      - Stage 2 readout decoder sweep (requires checkpoint)"
        echo "  optimizer    - Optimizer and LR schedule sweep"
        echo "  350m_encoder - Stage 1 350M encoder architecture sweep"
        echo "  all          - Run complete 180M hyperparameter study"
        echo "  generate     - Generate Slurm scripts without submitting"
        exit 1
        ;;
esac

echo ""
echo "Check job status with: squeue -u \$USER"
echo "View logs in: logs/"

