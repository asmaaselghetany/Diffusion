#!/usr/bin/env bash
# Shared block_qwen eval launcher (samples → gen-PPL → DepBench → ELBO sweep).
#
# Usage:
#   scripts/_block_qwen_eval.bash <checkpoint_path> [extra run_block_qwen_eval.py args...]
#
# Environment:
#   EVAL_RUN_DIR  — override output directory (default: <ckpt_run>/eval)

set -euo pipefail

CKPT="${1:?Usage: _block_qwen_eval.bash <checkpoint> [eval args...]}"
shift

WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

module load CUDA/12.6.0 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/_block_qwen_ckpt.bash

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
# DepBench is a sibling package; editable install is brittle here (broken
# uni-d2 file: URL in its pyproject). Always put the repo root on PYTHONPATH.
export DEPBENCH_ROOT="${DEPBENCH_ROOT:-${WORKSPACE}/projects/depbench}"
export PYTHONPATH="${DEPBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CKPT_ABS="$(readlink -f "${CKPT}")"
CKPT_BASE="$(basename "${CKPT_ABS}")"
CKPT_DIR="$(dirname "${CKPT_ABS}")"
# last.ckpt is often stale while last-v1 holds the real weights. When the caller
# asks for last*.ckpt, heal to the highest-step valid file first.
if [[ "${CKPT_BASE}" == last.ckpt || "${CKPT_BASE}" == last-v*.ckpt ]]; then
  echo "eval: preparing max-step last.ckpt under ${CKPT_DIR}"
  CKPT_ABS="$(readlink -f "$(_block_qwen_prepare_last_ckpt "${CKPT_DIR}")")"
fi
STEP="$(_block_qwen_ckpt_global_step "${CKPT_ABS}" || echo -1)"
RUN_DIR="${EVAL_RUN_DIR:-$(dirname "$(dirname "${CKPT_ABS}")")/eval}"
mkdir -p "${RUN_DIR}" slurm_logs

echo "=== block_qwen eval ==="
echo "  checkpoint: ${CKPT_ABS}"
echo "  global_step:${STEP}"
echo "  run_dir:    ${RUN_DIR}"

srun python -u tools/run_block_qwen_eval.py \
  --checkpoint "${CKPT_ABS}" \
  --run-dir "${RUN_DIR}" \
  "$@"

echo "Done. Results under ${RUN_DIR}"
