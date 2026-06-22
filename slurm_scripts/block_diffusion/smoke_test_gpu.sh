#!/usr/bin/env bash
# Short GPU smoke on one Booster GH200 (optional, after smoke_test_login.sh).
#
# Runs 2 training steps with flex attention on CUDA — catches triton/runtime issues
# that login CPU checks cannot exercise.
#
# Usage:
#   bash slurm_scripts/block_diffusion/smoke_test_gpu.sh
#   MODE=uniform bash slurm_scripts/block_diffusion/smoke_test_gpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jupiter_paths.sh
source "${SCRIPT_DIR}/jupiter_paths.sh"

REPO_ROOT="${JEDI_COMPUTE_ROOT}"
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  echo "ERROR: Compute workspace missing at ${REPO_ROOT}" >&2
  echo "Run on login: bash slurm_scripts/block_diffusion/setup_compute_workspace.sh" >&2
  exit 1
fi

MODE="${MODE:-masked}"
LOG_DIR="${REPO_ROOT}/outputs/block_diffusion/logs"
mkdir -p "${LOG_DIR}"

JOB_ID=$(sbatch --parsable \
  --job-name=bd_smoke_gpu \
  --partition=booster \
  --nodes=1 \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  --cpus-per-task=18 \
  --time=00:10:00 \
  --chdir="${REPO_ROOT}" \
  --output="${LOG_DIR}/smoke_gpu_%j.out" \
  --error="${LOG_DIR}/smoke_gpu_%j.err" \
  --wrap="bash -lc '
    set -euo pipefail
    export MODE=${MODE}
    export DATA=tiny_shakespeare
    export ATTN_BACKEND=flex
    export SEQ_LEN=128
    export BLOCK_SIZE=16
    export GLOBAL_BATCH=8
    export MAX_STEPS=2
    export VAL_CHECK_INTERVAL=100000
    export NUM_WORKERS=2
    export NUM_NODES=1
    export GPUS_PER_NODE=1
    export WANDB_MODE=offline
    export REPO_ROOT=${REPO_ROOT}
    export PYTHONSTARTUP=${REPO_ROOT}/scripts/triton_sitecustomize.py
    bash slurm_scripts/block_diffusion/train_core.sh
  '")

echo "Submitted GPU smoke job ${JOB_ID}"
echo "  stdout: ${LOG_DIR}/smoke_gpu_${JOB_ID}.out"
echo "  stderr: ${LOG_DIR}/smoke_gpu_${JOB_ID}.err"
echo "Watch:  squeue -j ${JOB_ID}"
