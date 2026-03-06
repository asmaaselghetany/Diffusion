#!/usr/bin/env bash
set -euo pipefail

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

MODEL="${MODEL:-}"
if [[ -z "${MODEL}" ]]; then
  echo "ERROR: MODEL is required (mdlm | latent_jepa | continuous)." >&2
  exit 2
fi

case "${MODEL}" in
  mdlm|latent_jepa|continuous) ;;
  *)
    echo "ERROR: unsupported MODEL='${MODEL}'. Use: mdlm, latent_jepa, continuous." >&2
    exit 2
    ;;
esac

REPO_ROOT="${REPO_ROOT:-/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi}"
RUN_DIR="${RUN_DIR:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

NUM_SAMPLES="${NUM_SAMPLES:-64}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
EVAL_MODEL="${EVAL_MODEL:-gpt2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"

MODEL_TOKENIZER="${MODEL_TOKENIZER:-gpt2}"
SEQ_LEN="${SEQ_LEN:-128}"

case "${MODEL}" in
  mdlm)
    NUM_STEPS="${NUM_STEPS:-128}"
    ;;
  latent_jepa)
    NUM_STEPS="${NUM_STEPS:-64}"
    ;;
  continuous)
    NUM_STEPS="${NUM_STEPS:-64}"
    ;;
esac

MAX_LENGTH="${MAX_LENGTH:-${SEQ_LEN}}"

cd "${REPO_ROOT}"
set +u
source ~/.bashrc || true
set -u

CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV_NAME}" ]] && command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
else
  source venv/bin/activate 2>/dev/null || true
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

declare -a ATTEMPTED_PATHS=()
RESOLVED_CKPT=""

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  ATTEMPTED_PATHS+=("${CHECKPOINT_PATH}")
  if [[ -f "${CHECKPOINT_PATH}" ]]; then
    RESOLVED_CKPT="${CHECKPOINT_PATH}"
  fi
fi

if [[ -z "${RESOLVED_CKPT}" && -n "${RUN_DIR}" ]]; then
  BEST_CKPT="${RUN_DIR}/checkpoints/best.ckpt"
  LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"
  ATTEMPTED_PATHS+=("${BEST_CKPT}" "${LAST_CKPT}")

  if [[ -f "${BEST_CKPT}" ]]; then
    RESOLVED_CKPT="${BEST_CKPT}"
  elif [[ -f "${LAST_CKPT}" ]]; then
    RESOLVED_CKPT="${LAST_CKPT}"
  fi
fi

if [[ -z "${RESOLVED_CKPT}" ]]; then
  echo "ERROR: Could not resolve checkpoint." >&2
  if [[ ${#ATTEMPTED_PATHS[@]} -eq 0 ]]; then
    echo "Attempted paths: <none> (set CHECKPOINT_PATH or RUN_DIR)." >&2
  else
    echo "Attempted paths:" >&2
    for p in "${ATTEMPTED_PATHS[@]}"; do
      echo "  - ${p}" >&2
    done
  fi
  exit 2
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(cd "$(dirname "${RESOLVED_CKPT}")/.." && pwd)"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RUN_DIR}/eval/gen_ppl/${TIMESTAMP}"
SAMPLES_PATH="${OUT_DIR}/samples.pt"
METRICS_PATH="${OUT_DIR}/gen_ppl_metrics.json"
mkdir -p "${OUT_DIR}"

GEN_CMD=(
  "${PYTHON_CMD[@]}" -u -m discrete_diffusion.evaluations.generate_samples
  "checkpoint_path=${RESOLVED_CKPT}"
  "samples_path=${SAMPLES_PATH}"
  "num_samples=${NUM_SAMPLES}"
  "batch_size=${GEN_BATCH_SIZE}"
  "num_steps=${NUM_STEPS}"
  "device=${DEVICE}"
  "torch_compile=false"
  "save_text=true"
)

PPL_CMD=(
  "${PYTHON_CMD[@]}" -u -m discrete_diffusion.evaluations.generative_ppl
  "samples_path=${SAMPLES_PATH}"
  "model_tokenizer=${MODEL_TOKENIZER}"
  "pretrained_model=${EVAL_MODEL}"
  "batch_size=${EVAL_BATCH_SIZE}"
  "max_length=${MAX_LENGTH}"
  "retokenize=true"
  "first_chunk_only=true"
  "metrics_path=${METRICS_PATH}"
  "torch_compile=false"
)

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  GEN_CMD=(srun "${GEN_CMD[@]}")
  PPL_CMD=(srun "${PPL_CMD[@]}")
fi

echo "=========================================="
echo "Tiny Shakespeare Baseline Evaluation"
echo "  MODEL:          ${MODEL}"
echo "  RUN_DIR:        ${RUN_DIR}"
echo "  CHECKPOINT:     ${RESOLVED_CKPT}"
echo "  OUT_DIR:        ${OUT_DIR}"
echo "  NUM_SAMPLES:    ${NUM_SAMPLES}"
echo "  NUM_STEPS:      ${NUM_STEPS}"
echo "  EVAL_MODEL:     ${EVAL_MODEL}"
echo "  MODEL_TOKENIZER:${MODEL_TOKENIZER}"
echo "=========================================="

if is_true "${DRY_RUN}"; then
  echo "DRY_RUN=1 -> generate command:"
  printf ' %q' "${GEN_CMD[@]}"
  echo
  echo "DRY_RUN=1 -> ppl command:"
  printf ' %q' "${PPL_CMD[@]}"
  echo
  exit 0
fi

"${GEN_CMD[@]}"
"${PPL_CMD[@]}"

echo "=========================================="
echo "Evaluation finished"
echo "Samples: ${SAMPLES_PATH}"
echo "Metrics: ${METRICS_PATH}"
echo "=========================================="
