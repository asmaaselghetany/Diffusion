#!/bin/bash
#SBATCH --job-name=bqwen-lm-eval-smoke
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --chdir=/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2

# Smoke: Fast-dLLM-style task benches + tok/s on one arm.
#   CKPT=... LIMIT=2 TASKS=gsm8k sbatch scripts/slurm/lm_eval_smoke.sbatch
set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

CKPT="${CKPT:-outputs/block_qwen/ar2block_masked_139760/checkpoints/last.ckpt}"
export LIMIT="${LIMIT:-2}"
export TASKS="${TASKS:-gsm8k}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_STEPS="${NUM_STEPS:-8}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export THROUGHPUT_BATCHES="${THROUGHPUT_BATCHES:-2}"
export OUT_DIR="${OUT_DIR:-$(dirname "$(dirname "$(readlink -f "${CKPT}")")")/lm_eval_smoke}"

echo "CKPT=${CKPT}"
echo "OUT_DIR=${OUT_DIR} TASKS=${TASKS} LIMIT=${LIMIT}"
nvidia-smi -L || true

bash examples/block_qwen/lm_eval.sh "${CKPT}"

echo "=== SUMMARY ==="
cat "${OUT_DIR}/SUMMARY.md" || true
ls -la "${OUT_DIR}"
