#!/usr/bin/env bash
# Cross-stack probe: our BlockTrainer ckpt → Hub Fast-dLLM generate (batch_sample).
#
# Isolates weights/data vs generate-stack without changing the conversion
# baseline. Export runs in train venv; lm-eval uses Hub venv (transformers 4.53).
#
# Usage:
#   ./scripts/submit_cross_stack_hub_generate.sh <ckpt|run_dir>
#   TASKS=gsm8k,ifeval THRESHOLD=1 ./scripts/submit_cross_stack_hub_generate.sh <ckpt>
#   ./scripts/submit_cross_stack_hub_generate.sh <ckpt> --dry-run

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

DRY=0
CKPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
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

RUN_DIR="$(dirname "$(dirname "${CKPT}")")"
TAG="$(basename "${RUN_DIR}")"
# Keep export under hub_fastdllm_eval so submit_hub_fastdllm_eval OUT naming is clear.
EXPORT_DIR="${EXPORT_DIR:-${REPO_ROOT}/outputs/hub_fastdllm_eval/our_${TAG}}"
export THRESHOLD="${THRESHOLD:-1}"
export TASKS="${TASKS:-gsm8k,ifeval}"
export USE_BLOCK_CACHE="${USE_BLOCK_CACHE:-False}"
export NUM_NODES="${NUM_NODES:-4}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
# submit_hub_fastdllm_eval.sh builds OUT from basename(MODEL_PATH)_t${THRESHOLD}
export MODEL_PATH="${EXPORT_DIR}"

echo "=== cross-stack Hub generate ==="
echo "  ckpt:   ${CKPT}"
echo "  export: ${EXPORT_DIR}"
echo "  thr=${THRESHOLD} tasks=${TASKS} use_block_cache=${USE_BLOCK_CACHE}"

if [[ "${DRY}" -eq 1 ]]; then
  echo "DRY: would export then MODEL_PATH=${EXPORT_DIR} submit_hub_fastdllm_eval"
  exit 0
fi

if [[ ! -f "${EXPORT_DIR}/model.safetensors" || "${FORCE_EXPORT:-0}" == "1" ]]; then
  PYTHONPATH=src "${REPO_ROOT}/.venv/bin/python" \
    tools/export_block_ckpt_to_fastdllm_hf.py \
    --ckpt "${CKPT}" \
    --out "${EXPORT_DIR}" \
    --overwrite
else
  echo "Reusing existing export: ${EXPORT_DIR}"
fi

export SKIP_PREFETCH="${SKIP_PREFETCH:-0}"
exec ./scripts/submit_hub_fastdllm_eval.sh
