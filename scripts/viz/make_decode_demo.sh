#!/usr/bin/env bash
# Record + render decode highlight videos for one or more checkpoints.
#
# Usage:
#   ./scripts/viz/make_decode_demo.sh \\
#     --name baseline_masked \\
#     --ckpt outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt \\
#     --profile hubmatch --thr 1.0
#
#   ./scripts/viz/make_decode_demo.sh --compare \\
#     docs/research/decode_viz/baseline_masked \\
#     docs/research/decode_viz/baseline_uniform
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

OUT_ROOT="${OUT_ROOT:-docs/research/decode_viz}"
PROMPT="${PROMPT:-Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?}"
DEVICE="${DEVICE:-cuda}"
MAX_NEW="${MAX_NEW_TOKENS:-128}"
STEPS="${NUM_STEPS:-32}"

if [[ "${1:-}" == "--compare" ]]; then
  shift
  LEFT="${1:?left trace dir}"
  RIGHT="${2:?right trace dir}"
  CMP_OUT="${3:-${OUT_ROOT}/compare_$(basename "${LEFT}")_vs_$(basename "${RIGHT}")}"
  "${PY}" scripts/viz/compare_decode_traces.py \
    --left "${LEFT}" --right "${RIGHT}" --out "${CMP_OUT}"
  echo "Compare → ${CMP_OUT}/preview.gif"
  exit 0
fi

NAME=""
CKPT=""
PROFILE="baseline"
THR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --thr) THR="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${NAME}" && -n "${CKPT}" ]] || {
  echo "Need --name and --ckpt" >&2
  exit 2
}

TRACE_DIR="${OUT_ROOT}/${NAME}"
RENDER_DIR="${TRACE_DIR}/render"
mkdir -p "${TRACE_DIR}"

REC_ARGS=(
  --ckpt "${CKPT}"
  --out "${TRACE_DIR}"
  --label "${NAME}"
  --prompt "${PROMPT}"
  --decode-profile "${PROFILE}"
  --max-new-tokens "${MAX_NEW}"
  --num-steps "${STEPS}"
  --device "${DEVICE}"
)
if [[ -n "${THR}" ]]; then
  REC_ARGS+=(--unmask-threshold "${THR}")
fi

echo "Recording ${NAME} …"
"${PY}" scripts/viz/record_decode_trace.py "${REC_ARGS[@]}"
echo "Rendering ${NAME} …"
"${PY}" scripts/viz/render_decode_video.py \
  --trace "${TRACE_DIR}" --out "${RENDER_DIR}"
echo "Done → ${RENDER_DIR}/preview.gif  (${RENDER_DIR}/viewer.html)"
