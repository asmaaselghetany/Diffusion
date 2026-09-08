#!/usr/bin/env bash
# Family-aware eval submitter — closes the "which stack?" gap.
#
# Auto-routes from checkpoint hydra config:
#   masked + DualCache/confidence pins → chat lm-eval (DECODE_PROFILE=dual_cache …)
#   masked without those pins          → chat lm-eval baseline (C0 floor)
#   uniform / ARPC                     → eval_checkpoint + arpc_decode_eval
#
# Offline free-gen (via eval_checkpoint → run_block_qwen_eval) always uses
# sample_mode=auto + decode_profile=baseline unless you override those flags.
#
# Usage:
#   ./scripts/submit_family_eval.sh <ckpt>
#   ./scripts/submit_family_eval.sh <ckpt> --dry-run
#   ./scripts/submit_family_eval.sh <ckpt> --lm-eval-only
#   ./scripts/submit_family_eval.sh <ckpt> --offline-only
#   FORCE_STACK=fastdllm_lm_eval ./scripts/submit_family_eval.sh <ckpt>
#   # Override decode even on auto:
#   FORCE_DECODE_PROFILE=baseline ./scripts/submit_family_eval.sh <ckpt>

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh" 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/e/project1/scifi/elsayed3/Diffusion-new}"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source scripts/_infer_block_qwen_eval_profile.bash

DRY_RUN=0
LM_ONLY=0
OFFLINE_ONLY=0
CKPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --lm-eval-only) LM_ONLY=1; shift ;;
    --offline-only) OFFLINE_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    -*)
      echo "Unknown flag: $1" >&2
      exit 2
      ;;
    *)
      CKPT="$1"
      shift
      ;;
  esac
done

[[ -n "${CKPT}" ]] || { echo "Usage: $0 <ckpt|run_dir> [--dry-run]" >&2; exit 2; }
CKPT="$(readlink -f "${CKPT}")"
# Allow a run directory (resolves checkpoints/last.ckpt|best.ckpt).
if [[ -d "${CKPT}" ]]; then
  if [[ -f "${CKPT}/checkpoints/last.ckpt" ]]; then
    CKPT="$(readlink -f "${CKPT}/checkpoints/last.ckpt")"
  elif [[ -f "${CKPT}/checkpoints/best.ckpt" ]]; then
    CKPT="$(readlink -f "${CKPT}/checkpoints/best.ckpt")"
  else
    echo "Run dir has no checkpoints/last.ckpt|best.ckpt: ${CKPT}" >&2
    exit 1
  fi
fi
[[ -f "${CKPT}" ]] || { echo "Missing ckpt: ${CKPT}" >&2; exit 1; }

infer_block_qwen_eval_profile "${CKPT}"
STACK="${FORCE_STACK:-${EVAL_STACK}}"

echo "=== submit_family_eval ==="
echo "  ckpt:    ${CKPT}"
echo "  hydra:   ${EVAL_HYDRA_CFG}"
echo "  forward: ${EVAL_FORWARD}  family=${EVAL_FAMILY}"
echo "  stack:   ${STACK}"
echo "  reason:  ${EVAL_REASON}"

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY: $*"
  else
    "$@"
  fi
}

submit_lm_eval() {
  # Stack defaults are positional. Explicit FORCE_* wins; bare DECODE_PROFILE
  # from a prior shell must not silently demote fastdllm exactness → baseline.
  local profile="${FORCE_DECODE_PROFILE:-$1}"
  local thr="${FORCE_UNMASK_THRESHOLD:-$2}"
  local greedy="${FORCE_GREEDY_PIN:-$3}"
  export CKPT
  export DECODE_PROFILE="${profile}"
  export UNMASK_THRESHOLD="${thr}"
  export FORCE_GREEDY="${greedy}"
  export SUITE="${SUITE:-custom}"
  # Keep TASKS if caller set it; else partial paper suite that is prefetched.
  export TASKS="${TASKS:-gsm8k,ifeval,humaneval,humaneval_plus,mbpp,mbpp_plus}"
  export NUM_NODES="${NUM_NODES:-8}"
  export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
  export NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_NODES * GPUS_PER_NODE))}"
  export LM_EVAL_DIST_TIMEOUT_SEC="${LM_EVAL_DIST_TIMEOUT_SEC:-21600}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
  local job_name="${JOB_NAME:-bqwen-lm-eval}"
  echo "  lm_eval: profile=${DECODE_PROFILE} thr=${UNMASK_THRESHOLD:-none} greedy=${FORCE_GREEDY:-task} tasks=${TASKS} nodes=${NUM_NODES}x${GPUS_PER_NODE}"
  run sbatch --parsable --job-name="${job_name}" --nodes="${NUM_NODES}" --time="${TIME_LIMIT:-12:00:00}" \
    --export=ALL \
    scripts/slurm/lm_eval.sbatch
}

submit_offline() {
  echo "  offline: eval_checkpoint.sbatch"
  run sbatch --parsable --job-name="${JOB_NAME:-bqwen-eval}" \
    scripts/slurm/eval_checkpoint.sbatch "${CKPT}"
}

submit_arpc() {
  local run_dir
  run_dir="$(dirname "$(dirname "${CKPT}")")"
  local out="${ARPC_OUT:-${run_dir}/eval_arpc}"
  export ARPC_MODE="${ARPC_MODE:-${EVAL_ARPC_MODE:-blockgen}}"
  echo "  arpc: out=${out} mode=${ARPC_MODE}"
  run sbatch --parsable --job-name="${JOB_NAME:-bqwen-arpc}" --export=ALL \
    scripts/slurm/arpc_decode_eval.sbatch "${CKPT}" "${out}" "${NUM_SAMPLES:-8}"
}

case "${STACK}" in
  fastdllm_lm_eval)
    if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
      submit_offline
    else
      # Exactness defaults; caller can still override DECODE_PROFILE=baseline.
      id="$(submit_lm_eval dual_cache "${EVAL_UNMASK_THRESHOLD:-0.9}" 1)"
      echo "Submitted lm_eval job: ${id}"
      if [[ "${LM_ONLY}" -eq 0 ]]; then
        # Optional offline samples/ELBO alongside (uses ckpt pins).
        oid="$(submit_offline || true)"
        echo "Submitted offline eval job: ${oid}"
      fi
    fi
    ;;
  conversion_lm_eval)
    if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
      submit_offline
    else
      id="$(submit_lm_eval baseline "" "")"
      echo "Submitted lm_eval job: ${id}"
    fi
    ;;
  blockgen_arpc)
    if [[ "${LM_ONLY}" -eq 1 ]]; then
      echo "REFUSED: chat lm-eval is not the BlockGen/uniform stack." >&2
      echo "Use masked conversion ckpts, or omit --lm-eval-only." >&2
      exit 2
    fi
    oid="$(submit_offline)"
    echo "Submitted offline eval job: ${oid}"
    aid="$(submit_arpc)"
    echo "Submitted ARPC decode job: ${aid}"
    ;;
  *)
    echo "Unknown EVAL_STACK=${STACK} (set FORCE_STACK=...)" >&2
    exit 2
    ;;
esac
