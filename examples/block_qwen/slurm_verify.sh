#!/usr/bin/env bash
#SBATCH --job-name=block-qwen-verify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=block_qwen_verify_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH=src

bash examples/block_qwen/smoke.sh
