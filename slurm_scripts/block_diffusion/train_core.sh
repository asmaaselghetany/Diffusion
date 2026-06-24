#!/usr/bin/env bash
# Shared training launcher for BlockDiffusion on Jupiter Booster.
#
# Usage (interactive or inside an sbatch job):
#   MODE=masked ./slurm_scripts/block_diffusion/train_core.sh
#   MODE=uniform DATA=openwebtext-split ./slurm_scripts/block_diffusion/train_core.sh
#
# Booster allocates a full node per job — always use 4 GPUs per node.
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

MODE="${MODE:-masked}"
case "${MODE}" in
  masked) ALGO="block_diffusion_masked" ;;
  uniform) ALGO="block_diffusion_uniform" ;;
  *)
    echo "ERROR: MODE must be 'masked' or 'uniform', got '${MODE}'" >&2
    exit 2
    ;;
esac

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REPO_ROOT="${REPO_ROOT:-$(pwd)}"
  SCRIPT_DIR_CORE="${REPO_ROOT}/slurm_scripts/block_diffusion"
else
  SCRIPT_DIR_CORE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR_CORE}/../.." && pwd)}"
fi
# shellcheck source=jupiter_paths.sh
source "${SCRIPT_DIR_CORE}/jupiter_paths.sh"
REPO_ROOT="$(resolve_jedi_repo_root)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ ! -d "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(translate_scifi_path "${REPO_ROOT}")"
  fi
  if [[ -d "${JEDI_COMPUTE_ROOT}" ]]; then
    REPO_ROOT="${JEDI_COMPUTE_ROOT}"
  fi
  require_compute_visible_repo "${REPO_ROOT}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/block_diffusion}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_ROOT}/data_cache}"
WANDB_PROJECT="${WANDB_PROJECT:-block_diffusion}"
# Booster compute nodes have no outbound HTTP; use offline wandb unless overridden.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  WANDB_MODE="${WANDB_MODE:-offline}"
else
  WANDB_MODE="${WANDB_MODE:-online}"
fi
DRY_RUN="${DRY_RUN:-0}"

# Booster: one node = 4 GH200 GPUs (bill full node even if you request fewer).
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-1024}"
MAX_STEPS="${MAX_STEPS:-1000000}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
EVAL_GLOBAL_BATCH="${EVAL_GLOBAL_BATCH:-${GLOBAL_BATCH}}"
DATA="${DATA:-tiny_shakespeare}"
ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-100}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-10000}"
GENERATE_VAL_SAMPLES="${GENERATE_VAL_SAMPLES:-0}"
SAVE_VAL_SAMPLES="${SAVE_VAL_SAMPLES:-0}"
RESAMPLE="${RESAMPLE:-1}"
# MDLM pretrain for OWT block-diffusion finetune (BD3-LM Baseline B).
# Set FROM_PRETRAINED=1 or path to .ckpt; default uses data_cache/checkpoints/...
FROM_PRETRAINED="${FROM_PRETRAINED:-}"
# AR pretrain for AR → block-diffusion finetune (kuleshov-group/ar-noeos-owt).
FROM_AR_PRETRAINED="${FROM_AR_PRETRAINED:-}"

if [[ -n "${FROM_PRETRAINED:-}" ]] && [[ -n "${FROM_AR_PRETRAINED:-}" ]]; then
  echo "ERROR: Set only one of FROM_PRETRAINED or FROM_AR_PRETRAINED." >&2
  exit 2
fi

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
PER_DEVICE_BATCH=$((GLOBAL_BATCH / TOTAL_GPUS))
if [[ "${PER_DEVICE_BATCH}" -lt 1 ]]; then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} too small for ${TOTAL_GPUS} GPUs." >&2
  exit 2
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${ALGO}_b${BLOCK_SIZE}_${TIMESTAMP}}"
if [[ -n "${FROM_PRETRAINED:-}" ]]; then
  RUN_NAME="${RUN_NAME}_mdlminit"
elif [[ -n "${FROM_AR_PRETRAINED:-}" ]]; then
  RUN_NAME="${RUN_NAME}_arinit"
fi
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${DATA_CACHE_DIR}"

cd "${REPO_ROOT}"
set +u
source ~/.bashrc 2>/dev/null || true
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v jutil >/dev/null 2>&1; then
  jutil env activate -p scifi 2>/dev/null || true
fi
set -u

activate_jupiter_modules

CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV_NAME}" ]] && command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" --no-capture-output python)
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  # Module Python (3.13) can shadow venv after `activate` on compute — use explicit path.
  PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
  PYTHON_CMD=(python)
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if ! "${PYTHON_CMD[@]}" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable — run setup_cuda_venv.sh on login and sync to /e'" 2>/dev/null; then
    echo "ERROR: PyTorch cannot see CUDA on this node." >&2
    "${PYTHON_CMD[@]}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'built', torch.version.cuda)" 2>&1 || true
    echo "Fix on login: bash slurm_scripts/block_diffusion/setup_cuda_venv.sh" >&2
    echo "Then: bash slurm_scripts/block_diffusion/setup_compute_workspace.sh" >&2
    exit 1
  fi
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONSTARTUP="${REPO_ROOT}/scripts/triton_sitecustomize.py"
export WANDB_MODE
export BLOCK_SIZE
export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${DATA_CACHE_DIR}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${DATA_CACHE_DIR}/hf_hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

GENERATE_VAL_SAMPLES_BOOL="$(to_bool_literal "${GENERATE_VAL_SAMPLES}")"
SAVE_VAL_SAMPLES_BOOL="$(to_bool_literal "${SAVE_VAL_SAMPLES}")"
RESAMPLE_BOOL="$(to_bool_literal "${RESAMPLE}")"

TRAIN_ARGS=(
  "data=${DATA}"
  "data.cache_dir=${DATA_CACHE_DIR}"
  "model=block_dit"
  "model.length=${SEQ_LEN}"
  "algo=${ALGO}"
  "sampling=block_diffusion"
  "block_size=${BLOCK_SIZE}"
  "strategy=ddp"
  "loader.global_batch_size=${GLOBAL_BATCH}"
  "loader.eval_global_batch_size=${EVAL_GLOBAL_BATCH}"
  "loader.batch_size=${PER_DEVICE_BATCH}"
  "loader.eval_batch_size=${PER_DEVICE_BATCH}"
  "loader.num_workers=${NUM_WORKERS}"
  "trainer.num_nodes=${NUM_NODES}"
  "trainer.devices=${GPUS_PER_NODE}"
  "trainer.accelerator=cuda"
  "trainer.max_steps=${MAX_STEPS}"
  "trainer.precision=${PRECISION}"
  "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
  "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}"
  "trainer.num_sanity_val_steps=2"
  "training.resample=${RESAMPLE_BOOL}"
  "algo.ignore_bos=True"
  "model.tie_word_embeddings=True"
  "model.attn_backend=${ATTN_BACKEND}"
  "callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS}"
  "callbacks.checkpoint_every_n_steps.save_top_k=-1"
  "callbacks.checkpoint_every_n_steps.save_last=true"
  "callbacks.checkpoint_monitor.save_top_k=1"
  "eval.generate_samples=${GENERATE_VAL_SAMPLES_BOOL}"
  "eval.save_validation_samples=${SAVE_VAL_SAMPLES_BOOL}"
  "checkpointing.save_dir=${RUN_DIR}"
  "checkpointing.resume_from_ckpt=false"
  "wandb.project=${WANDB_PROJECT}"
  "wandb.name=${RUN_NAME}"
  "wandb.save_dir=${RUN_DIR}/wandb"
  "hydra.run.dir=${RUN_DIR}"
)

# Shared data settings for BD3-LM-aligned pretrain finetuning (MDLM + AR).
_pretrain_data_args() {
  TRAIN_ARGS+=(
    "data.insert_train_eos=False"
    "data.insert_valid_eos=False"
    "data.insert_train_special=False"
    "data.insert_valid_special=False"
    "model.tie_word_embeddings=False"
    "algo.cross_attn=False"
  )
}

# BD3-LM OWT MDLM pretrain (kuleshov-group/bd3lm-owt-block_size1024-pretrain)
if [[ -n "${FROM_PRETRAINED:-}" ]]; then
  PRETRAIN_CKPT="${FROM_PRETRAINED}"
  if [[ "${PRETRAIN_CKPT}" == "1" ]] || [[ "${PRETRAIN_CKPT}" == "true" ]] || [[ "${PRETRAIN_CKPT}" == "default" ]]; then
    PRETRAIN_CKPT="${DATA_CACHE_DIR}/checkpoints/bd3lm_owt_block1024_pretrain.ckpt"
  fi
  if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "ERROR: MDLM pretrain checkpoint not found: ${PRETRAIN_CKPT}" >&2
    echo "On login node run: bash slurm_scripts/block_diffusion/download_mdlm_pretrain.sh" >&2
    exit 1
  fi
  if [[ "${SEQ_LEN}" != "1024" ]]; then
    echo "WARNING: MDLM pretrain is for seq_len=1024 (OWT); got SEQ_LEN=${SEQ_LEN}" >&2
  fi
  TRAIN_ARGS+=(
    "training.from_pretrained=${PRETRAIN_CKPT}"
    "training.pretrain_profile=bd3lm"
    "model.adaln=True"
    "model.causal_attention=False"
  )
  _pretrain_data_args
elif [[ -n "${FROM_AR_PRETRAINED:-}" ]]; then
  PRETRAIN_CKPT="${FROM_AR_PRETRAINED}"
  if [[ "${PRETRAIN_CKPT}" == "1" ]] || [[ "${PRETRAIN_CKPT}" == "true" ]] || [[ "${PRETRAIN_CKPT}" == "default" ]]; then
    PRETRAIN_CKPT="${DATA_CACHE_DIR}/checkpoints/ar_noeos_owt.ckpt"
  fi
  if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "ERROR: AR pretrain checkpoint not found: ${PRETRAIN_CKPT}" >&2
    echo "On login node run: bash slurm_scripts/block_diffusion/download_ar_pretrain.sh" >&2
    exit 1
  fi
  if [[ "${SEQ_LEN}" != "1024" ]]; then
    echo "WARNING: AR pretrain is for seq_len=1024 (OWT); got SEQ_LEN=${SEQ_LEN}" >&2
  fi
  TRAIN_ARGS+=(
    "training.from_pretrained=${PRETRAIN_CKPT}"
    "training.pretrain_profile=ar"
    "model.adaln=False"
    "model.causal_attention=True"
  )
  _pretrain_data_args
else
  TRAIN_ARGS+=(
    "training.pretrain_profile=block_diffusion"
    "model.adaln=False"
    "model.causal_attention=False"
    "algo.cross_attn=True"
  )
fi

TORCHRUN_CMD=("${PYTHON_CMD[@]}" -m torch.distributed.run)

launch_training() {
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v "${TORCHRUN_CMD[0]}" >/dev/null 2>&1; then
    local master_node master_addr
    master_node="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
    master_addr="$(srun --nodes=1 --ntasks=1 -w "${master_node}" hostname -i 2>/dev/null | awk '{print $1}')"
    [[ -z "${master_addr}" ]] && master_addr=127.0.0.1
    export MASTER_ADDR="${master_addr}" MASTER_PORT="${MASTER_PORT:-29500}"
    echo "torchrun master: ${MASTER_ADDR}:${MASTER_PORT}"

    # Match stage2 SLURM launchers: torchrun spawns ranks; avoid Lightning SLURM mismatch.
    srun --kill-on-bad-exit=1 --export=ALL bash -c "
      $(printf '%q ' "${TORCHRUN_CMD[@]}") \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --node_rank=\${SLURM_NODEID:-0} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        -m discrete_diffusion $(printf '%q ' "${TRAIN_ARGS[@]}")
    "
  else
    "${PYTHON_CMD[@]}" -u -m discrete_diffusion "${TRAIN_ARGS[@]}"
  fi
}

echo "=========================================="
echo "BlockDiffusion training"
echo "  MODE:          ${MODE} (${ALGO})"
echo "  DATA:          ${DATA}"
echo "  RUN_NAME:      ${RUN_NAME}"
echo "  RUN_DIR:       ${RUN_DIR}"
echo "  NODES:         ${NUM_NODES}"
echo "  GPUS/node:     ${GPUS_PER_NODE} (total ${TOTAL_GPUS})"
echo "  GLOBAL_BATCH:  ${GLOBAL_BATCH} (${PER_DEVICE_BATCH}/GPU)"
echo "  BLOCK_SIZE:    ${BLOCK_SIZE}"
echo "  SEQ_LEN:       ${SEQ_LEN}"
echo "  MAX_STEPS:     ${MAX_STEPS}"
if [[ -n "${FROM_PRETRAINED:-}" ]]; then
  echo "  FROM_PRETRAINED:    ${PRETRAIN_CKPT} (MDLM, adaln=True)"
elif [[ -n "${FROM_AR_PRETRAINED:-}" ]]; then
  echo "  FROM_AR_PRETRAINED: ${PRETRAIN_CKPT} (AR, adaln=False)"
fi
echo "=========================================="

if is_true "${DRY_RUN}"; then
  echo "DRY_RUN=1 -> would run launch_training (torchrun on SLURM)"
  printf ' %q' "${PYTHON_CMD[@]}" -u -m discrete_diffusion
  printf ' %q' "${TRAIN_ARGS[@]}"
  echo
  exit 0
fi

launch_training

echo "=========================================="
echo "Training finished: ${RUN_DIR}"
echo "=========================================="
