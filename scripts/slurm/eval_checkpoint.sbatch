#!/usr/bin/env bash
#SBATCH --job-name=bqwen-eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Full suite (samples + DepBench grid) needs ~10-14h; 8h was timing out.
#SBATCH --partition=standard
#SBATCH --chdir=/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2
#
# Full eval suite for one block_qwen checkpoint.
#
# Usage:
#   sbatch scripts/slurm/eval_checkpoint.sbatch <checkpoint_path> [eval args...]
#   sbatch scripts/slurm/eval_checkpoint.sbatch masked <job_id> [eval args...]
#   sbatch scripts/slurm/eval_checkpoint.sbatch ar2block masked <job_id> [eval args...]
#   sbatch scripts/slurm/eval_checkpoint.sbatch block uniform <job_id> [eval args...]
#
# DepBench-only (reuse existing samples/gen-PPL/ELBO):
#   sbatch scripts/slurm/eval_checkpoint.sbatch <ckpt> \
#     --skip-samples --skip-gen-ppl --skip-elbo

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash
# shellcheck disable=SC1091
source scripts/_resolve_block_qwen_run.bash

ARG1="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"

if [[ -f "${ARG1}" ]]; then
  CKPT="${ARG1}"
  shift
elif [[ -n "${ARG1}" && "${ARG1}" =~ ^(ar2block|block|blockgen|neutral|auto)$ && -n "${ARG2}" && -n "${ARG3}" ]]; then
  CKPT="$(resolve_block_qwen_ckpt "${ARG1}" "${ARG2}" "${ARG3}")"
  shift 3
elif [[ -n "${ARG1}" && -n "${ARG2}" ]]; then
  CKPT="$(resolve_block_qwen_ckpt "${ARG1}" "${ARG2}")"
  shift 2
else
  echo "Usage: eval_checkpoint.sbatch <ckpt|arm job_id|line arm job_id> [eval args...]" >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_eval.bash "${CKPT}" "$@"
