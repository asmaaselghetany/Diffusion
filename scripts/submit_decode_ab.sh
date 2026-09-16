#!/usr/bin/env bash
# Decode-pack A/B for one conversion ckpt: hubmatch + DualCache (thr=1).
# Use for C0 vs C2eb fair floor under the same enhanced skeleton.
#
# Usage:
#   ./scripts/submit_decode_ab.sh /path/to/last.ckpt
#   TASKS=gsm8k,ifeval ./scripts/submit_decode_ab.sh <ckpt>
#   ./scripts/submit_decode_ab.sh <ckpt> --dry-run

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh" 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/e/project1/scifi/elsayed3/Diffusion-new}"
cd "${REPO_ROOT}"

DRY=0
CKPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    -*) echo "Unknown: $1" >&2; exit 2 ;;
    *) CKPT="$1"; shift ;;
  esac
done
[[ -n "${CKPT}" ]] || { echo "Usage: $0 <ckpt|run_dir>" >&2; exit 2; }
CKPT="$(readlink -f "${CKPT}")"
if [[ -d "${CKPT}" ]]; then
  if [[ -f "${CKPT}/checkpoints/last.ckpt" ]]; then
    CKPT="$(readlink -f "${CKPT}/checkpoints/last.ckpt")"
  elif [[ -f "${CKPT}/checkpoints/best.ckpt" ]]; then
    CKPT="$(readlink -f "${CKPT}/checkpoints/best.ckpt")"
  else
    echo "No checkpoints/last.ckpt|best.ckpt under ${CKPT}" >&2
    exit 1
  fi
fi
[[ -f "${CKPT}" ]] || { echo "Missing ckpt: ${CKPT}" >&2; exit 1; }

export SUITE="${SUITE:-paper_acc}"
export TASKS="${TASKS:-gsm8k,ifeval}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-8192}"
export NUM_NODES="${NUM_NODES:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

run_one() {
  local profile="$1"
  local thr="$2"
  export CKPT
  export FORCE_STACK=fastdllm_lm_eval
  export FORCE_DECODE_PROFILE="${profile}"
  export FORCE_UNMASK_THRESHOLD="${thr}"
  export FORCE_GREEDY_PIN=1
  export EVAL_MAX_SEQ_LEN
  if [[ "${DRY}" -eq 1 ]]; then
    echo "DRY: FORCE_DECODE_PROFILE=${profile} thr=${thr} EVAL_MAX_SEQ_LEN=${EVAL_MAX_SEQ_LEN} submit_family_eval --lm-eval-only"
    OUT_DIR= CKPT="${CKPT}" ./scripts/submit_family_eval.sh "${CKPT}" --lm-eval-only --dry-run || true
  else
    ./scripts/submit_family_eval.sh "${CKPT}" --lm-eval-only
  fi
}

echo "=== decode A/B (hubmatch + dual_cache) ==="
echo "  ckpt: ${CKPT}"
echo "  EVAL_MAX_SEQ_LEN=${EVAL_MAX_SEQ_LEN} MAX_NEW=${MAX_NEW_TOKENS}"
run_one hubmatch 1
run_one dual_cache 1
