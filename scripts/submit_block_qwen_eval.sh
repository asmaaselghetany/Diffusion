#!/usr/bin/env bash
# Submit eval jobs for block_qwen checkpoints.
#
# Usage:
#   ./scripts/submit_block_qwen_eval.sh all              # masked + uniform runs with ckpts
#   ./scripts/submit_block_qwen_eval.sh runs             # same as all
#   ./scripts/submit_block_qwen_eval.sh <checkpoint>     # single explicit path

set -euo pipefail

WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source scripts/_block_qwen_ckpt.bash
# shellcheck disable=SC1091
source .venv/bin/activate

submit_ckpt() {
  local ckpt="$1"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skip (missing): ${ckpt}"
    return 0
  fi
  local base dir prepared
  base="$(basename "${ckpt}")"
  dir="$(dirname "${ckpt}")"
  if [[ "${base}" == last.ckpt || "${base}" == last-v*.ckpt ]]; then
    prepared="$(_block_qwen_prepare_last_ckpt "${dir}")" || {
      echo "Skip (prepare failed): ${dir}" >&2
      return 0
    }
    ckpt="${prepared}"
  fi
  echo "Submitting eval for ${ckpt} (step=$(_block_qwen_ckpt_global_step "${ckpt}" || echo '?'))"
  sbatch scripts/slurm/eval_checkpoint.sbatch "${ckpt}"
}

submit_arm() {
  local arm="$1"
  local patterns=(
    "${REPO_ROOT}/outputs/block_qwen/${arm}_*"
    "${REPO_ROOT}/outputs/block_qwen/ar2block_${arm}_*"
    "${REPO_ROOT}/outputs/block_qwen/block_${arm}_*"
    "${REPO_ROOT}/outputs/block_qwen/blockgen_${arm}_*"
  )
  local found=0
  for pattern in "${patterns[@]}"; do
    for run_dir in ${pattern}; do
      [[ -d "${run_dir}" ]] || continue
      found=1
      # Prefer prepare over a possibly-stale last.ckpt path.
      if [[ -d "${run_dir}/checkpoints" ]]; then
        submit_ckpt "$(_block_qwen_prepare_last_ckpt "${run_dir}/checkpoints")"
      else
        submit_ckpt "${run_dir}/checkpoints/last.ckpt"
      fi
    done
  done
  if [[ "${found}" -eq 0 ]]; then
    echo "No ${arm} runs found under outputs/block_qwen/"
  fi
}

TARGET="${1:-all}"

if [[ -f "${TARGET}" ]]; then
  submit_ckpt "${TARGET}"
  exit 0
fi

case "${TARGET}" in
  runs|all)
    submit_arm masked
    submit_arm uniform
    ;;
  *)
    echo "Usage: $0 {all|runs|<checkpoint_path>}" >&2
    exit 1
    ;;
esac
