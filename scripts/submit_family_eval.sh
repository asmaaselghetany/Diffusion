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
  export SUITE="${SUITE:-paper_acc}"
  # Hub Fast-dLLM v2/eval.py parity (paper accuracy tables).
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
  # Hub has ~32k context; extend gen buffer beyond train length=2048 when needed.
  export EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-8192}"
  export NUM_STEPS="${NUM_STEPS:-32}"
  # Empty TASKS → lm_eval.sh expands full Fast-dLLM suite from SUITE.
  # Callers may still set TASKS=... for partial / resume runs.
  if [[ -z "${TASKS:-}" ]]; then
    unset TASKS || true
  else
    export TASKS
  fi
  # Tag OUT_DIR so thr/max_new cells do not clobber older lm_eval/ (512) trees.
  if [[ -z "${OUT_DIR:-}" ]]; then
    local run_dir
    run_dir="$(dirname "$(dirname "${CKPT}")")"
    local tag="m${MAX_NEW_TOKENS}_${profile}"
    if [[ -n "${thr}" ]]; then
      tag="${tag}_t${thr}"
    fi
    export OUT_DIR="${run_dir}/lm_eval_${tag}"
  else
    export OUT_DIR
  fi
  export NUM_NODES="${NUM_NODES:-8}"
  export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
  export NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_NODES * GPUS_PER_NODE))}"
  export LM_EVAL_DIST_TIMEOUT_SEC="${LM_EVAL_DIST_TIMEOUT_SEC:-21600}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
  local job_name="${JOB_NAME:-bqwen-lm-eval}"
  echo "  lm_eval: profile=${DECODE_PROFILE} thr=${UNMASK_THRESHOLD:-none} greedy=${FORCE_GREEDY:-task} max_new=${MAX_NEW_TOKENS} suite=${SUITE} tasks=${TASKS:-<suite default>} out=${OUT_DIR} nodes=${NUM_NODES}x${GPUS_PER_NODE}"
  # Never propagate a parent job's /tmp/block_qwen_<jid> into the eval allocation.
  run env -u TMPDIR -u TEMP -u TMP \
    sbatch --parsable --job-name="${job_name}" --nodes="${NUM_NODES}" --time="${TIME_LIMIT:-12:00:00}" \
    --export=ALL \
    scripts/slurm/lm_eval.sbatch
}

submit_offline() {
  echo "  offline: eval_checkpoint.sbatch"
  run env -u TMPDIR -u TEMP -u TMP \
    sbatch --parsable --job-name="${JOB_NAME:-bqwen-eval}" \
    scripts/slurm/eval_checkpoint.sbatch "${CKPT}"
}

submit_arpc() {
  local run_dir
  run_dir="$(dirname "$(dirname "${CKPT}")")"
  local out="${ARPC_OUT:-${run_dir}/eval_arpc}"
  export ARPC_MODE="${ARPC_MODE:-${EVAL_ARPC_MODE:-blockgen}}"
  echo "  arpc: out=${out} mode=${ARPC_MODE}"
  run env -u TMPDIR -u TEMP -u TMP \
    sbatch --parsable --job-name="${JOB_NAME:-bqwen-arpc}" --export=ALL \
    scripts/slurm/arpc_decode_eval.sbatch "${CKPT}" "${out}" "${NUM_SAMPLES:-8}"
}

case "${STACK}" in
  fastdllm_lm_eval)
    if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
      submit_offline
    else
      # Paper §4 accuracy default: DualCache + thr=1 (parallel decode off).
      # Do NOT inherit ckpt-baked thr=0.9 via EVAL_UNMASK_THRESHOLD.
      # Override with FORCE_UNMASK_THRESHOLD=0.9, or PAPER_BOTH_THR=1.
      id="$(submit_lm_eval dual_cache 1 1)"
      echo "Submitted lm_eval job (paper thr): ${id}"
      if [[ "${PAPER_BOTH_THR:-0}" == "1" ]]; then
        _saved_out="${OUT_DIR:-}"
        unset OUT_DIR || true
        JOB_NAME="${JOB_NAME:-bqwen-lm-eval}-thr09" \
          FORCE_UNMASK_THRESHOLD=0.9 \
          id09="$(submit_lm_eval dual_cache 0.9 1)"
        echo "Submitted lm_eval job (Hub thr=0.9): ${id09}"
        if [[ -n "${_saved_out}" ]]; then export OUT_DIR="${_saved_out}"; fi
      fi
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
