#!/usr/bin/env bash
#SBATCH --job-name=bqwen-genppl
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Re-run gen-PPL (+ write manifest) on an existing checkpoint that already has
# samples.pt / ELBO. Skips sample regen and DepBench.
#
# Usage:
#   sbatch scripts/slurm/gen_ppl_rerun.sbatch <checkpoint_path>
#   sbatch scripts/slurm/gen_ppl_rerun.sbatch <checkpoint_path> <eval_run_dir>

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

CKPT="${1:?Usage: gen_ppl_rerun.sbatch <checkpoint> [eval_run_dir]}"
EVAL_DIR="${2:-}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

EXTRA=()
if [[ -n "${EVAL_DIR}" ]]; then
  export EVAL_RUN_DIR="${EVAL_DIR}"
  mkdir -p "${EVAL_RUN_DIR}"
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_eval.bash "${CKPT}" \
  --skip-samples --skip-elbo --skip-depbench \
  "${EXTRA[@]}"
