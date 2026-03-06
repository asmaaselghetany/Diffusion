#!/usr/bin/env bash
set -euo pipefail

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

to_bool_literal() {
  if is_true "${1:-}"; then
    echo "true"
  else
    echo "false"
  fi
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
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/tiny_shakespeare/baselines}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/datasets/}"
WANDB_PROJECT="${WANDB_PROJECT:-tiny_shakespeare_baselines}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-500}"
GENERATE_VAL_SAMPLES="${GENERATE_VAL_SAMPLES:-1}"
SAVE_VAL_SAMPLES="${SAVE_VAL_SAMPLES:-1}"
VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-16}"
VAL_SAMPLE_BATCHES="${VAL_SAMPLE_BATCHES:-1}"
VAL_SAMPLE_LOG="${VAL_SAMPLE_LOG:-4}"
EVAL_GLOBAL_BATCH="${EVAL_GLOBAL_BATCH:-16}"
GENERATE_VAL_SAMPLES_BOOL="$(to_bool_literal "${GENERATE_VAL_SAMPLES}")"
SAVE_VAL_SAMPLES_BOOL="$(to_bool_literal "${SAVE_VAL_SAMPLES}")"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${MODEL}_${TIMESTAMP}}"
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"

mkdir -p "${RUN_DIR}" "${DATA_CACHE_DIR}" "${RUN_DIR}/wandb"

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
export WANDB_MODE
export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${DATA_CACHE_DIR}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${DATA_CACHE_DIR}/hf_hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

MODEL_OVERRIDES=()

case "${MODEL}" in
  mdlm)
    MAX_STEPS="${MAX_STEPS:-20000}"
    GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
    SEQ_LEN="${SEQ_LEN:-128}"
    MODEL_OVERRIDES=(
      "algo=mdlm"
      "model=small"
    )
    ;;

  latent_jepa)
    MAX_STEPS="${MAX_STEPS:-20000}"
    GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
    SEQ_LEN="${SEQ_LEN:-128}"
    MODEL_OVERRIDES=(
      "algo=jepa"
      "model=latent_jepa_small"
      "algo.stage=1"
      "callbacks.latent_eval.eval_frequency=50000"
      "strategy=single-device"
      "trainer.precision=16-mixed"
    )
    ;;

  continuous)
    MAX_STEPS="${MAX_STEPS:-10000}"
    GLOBAL_BATCH="${GLOBAL_BATCH:-16}"
    SEQ_LEN="${SEQ_LEN:-128}"

    CED_LAMBDA_CE="${CED_LAMBDA_CE:-0.1}"
    CED_PARAMETERIZATION="${CED_PARAMETERIZATION:-x0}"
    CED_LOSS_TYPE="${CED_LOSS_TYPE:-simple}"
    CED_HIDDEN_SIZE="${CED_HIDDEN_SIZE:-512}"
    CED_N_HEADS="${CED_N_HEADS:-8}"
    CED_N_BLOCKS="${CED_N_BLOCKS:-6}"
    CED_DECODER_N_BLOCKS="${CED_DECODER_N_BLOCKS:-2}"
    CED_ENCODER_NAME="${CED_ENCODER_NAME:-Qwen/Qwen3-Embedding-0.6B}"
    CED_ENCODER_DTYPE="${CED_ENCODER_DTYPE:-bfloat16}"

    MODEL_OVERRIDES=(
      "algo=continuous_embedding_diffusion"
      "model=continuous_embedding"
      "noise=cosine"
      "sampling=continuous_embedding"
      "algo.stage=1"
      "algo.parameterization=${CED_PARAMETERIZATION}"
      "algo.loss_type=${CED_LOSS_TYPE}"
      "algo.lambda_ce=${CED_LAMBDA_CE}"
      "algo.embedding_provider=legacy_contextual"
      "model.hidden_size=${CED_HIDDEN_SIZE}"
      "model.n_heads=${CED_N_HEADS}"
      "model.n_blocks=${CED_N_BLOCKS}"
      "model.decoder_n_blocks=${CED_DECODER_N_BLOCKS}"
      "model.encoder_name=${CED_ENCODER_NAME}"
      "model.encoder_dtype=${CED_ENCODER_DTYPE}"
      "strategy.find_unused_parameters=true"
    )
    ;;
esac

COMMON_OVERRIDES=(
  "data=tiny_shakespeare"
  "data.cache_dir=${DATA_CACHE_DIR}"
  "model.length=${SEQ_LEN}"
  "loader.global_batch_size=${GLOBAL_BATCH}"
  "loader.eval_global_batch_size=${EVAL_GLOBAL_BATCH}"
  "loader.num_workers=${NUM_WORKERS}"
  "trainer.num_nodes=${NUM_NODES}"
  "trainer.devices=${NUM_GPUS}"
  "trainer.max_steps=${MAX_STEPS}"
  "trainer.precision=${PRECISION}"
  "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
  "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}"
  "callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS}"
  "callbacks.checkpoint_every_n_steps.save_top_k=-1"
  "callbacks.checkpoint_every_n_steps.save_last=true"
  "callbacks.checkpoint_monitor.save_top_k=1"
  "callbacks.sample_saver.enabled=false"
  "eval.generate_samples=${GENERATE_VAL_SAMPLES_BOOL}"
  "eval.save_validation_samples=${SAVE_VAL_SAMPLES_BOOL}"
  "sampling.steps=${VAL_SAMPLE_STEPS}"
  "sampling.num_sample_batches=${VAL_SAMPLE_BATCHES}"
  "sampling.num_sample_log=${VAL_SAMPLE_LOG}"
  "checkpointing.save_dir=${RUN_DIR}"
  "checkpointing.resume_from_ckpt=false"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.name=${RUN_NAME}"
  "wandb.save_dir=${RUN_DIR}/wandb"
  "hydra.run.dir=${RUN_DIR}"
)

CMD=("${PYTHON_CMD[@]}" -u -m discrete_diffusion)
CMD+=("${COMMON_OVERRIDES[@]}")
CMD+=("${MODEL_OVERRIDES[@]}")

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  CMD=(srun "${CMD[@]}")
fi

echo "=========================================="
echo "Tiny Shakespeare Baseline Training"
echo "  MODEL:       ${MODEL}"
echo "  RUN_NAME:    ${RUN_NAME}"
echo "  RUN_DIR:     ${RUN_DIR}"
echo "  MAX_STEPS:   ${MAX_STEPS}"
echo "  GLOBAL_BATCH:${GLOBAL_BATCH}"
echo "  SEQ_LEN:     ${SEQ_LEN}"
echo "=========================================="

if is_true "${DRY_RUN}"; then
  echo "DRY_RUN=1 -> command:"
  printf ' %q' "${CMD[@]}"
  echo
  exit 0
fi

"${CMD[@]}"

echo "=========================================="
echo "Training finished"
echo "Run dir: ${RUN_DIR}"
echo "=========================================="
