#!/usr/bin/env bash
#SBATCH --job-name=bqwen-lm-eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=standard
#SBATCH --chdir=/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2
#
# Conditional lm-eval + optional tok/s for one checkpoint.
#
#   CKPT=outputs/.../last.ckpt sbatch scripts/slurm/lm_eval.sbatch
#   TASKS=gsm8k,ifeval CKPT=... sbatch scripts/slurm/lm_eval.sbatch

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

CKPT="${CKPT:?Set CKPT=/path/to.ckpt}"
CKPT="$(readlink -f "${CKPT}")"
[[ -f "${CKPT}" ]] || { echo "Missing ${CKPT}" >&2; exit 1; }

export TASKS="${TASKS:-gsm8k,ifeval}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export NUM_STEPS="${NUM_STEPS:-32}"
export SKIP_THROUGHPUT="${SKIP_THROUGHPUT:-0}"

echo "=== lm_eval ==="
echo "  ckpt:   ${CKPT}"
echo "  tasks:  ${TASKS}"
echo "  steps:  ${NUM_STEPS}"

bash examples/block_qwen/lm_eval.sh "${CKPT}"
