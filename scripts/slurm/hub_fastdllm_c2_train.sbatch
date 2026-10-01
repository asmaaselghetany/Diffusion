#!/usr/bin/env bash
#SBATCH --job-name=hub-fdllm-c2-tr
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Hub Fast-dLLM LMFlow train on our C2 Nemotron mix (external correctness check).
# Init = Qwen2.5-1.5B-Instruct weights in Fast_dLLM arch (AR→block), not continue-FT
# of the public Hub checkpoint (override with MODEL_PATH).

set -euo pipefail

export ASMAA_WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${ASMAA_WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/jupiter_paths.sh"
activate_jupiter_modules

# Prefer Stages CUDA nvcc for DeepSpeed import checks (ops built with DS_BUILD_OPS=0).
if [[ -x /e/software/default/stages/2026/software/CUDA/13/bin/nvcc ]]; then
  export CUDA_HOME="${CUDA_HOME:-/e/software/default/stages/2026/software/CUDA/13}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
elif [[ -x /e/software/default/stages/2025/software/CUDA/12/bin/nvcc ]]; then
  export CUDA_HOME="${CUDA_HOME:-/e/software/default/stages/2025/software/CUDA/12}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

HUB_TRAIN_VENV="${HUB_TRAIN_VENV:-/e/scratch/scifi/elsayed3/.venvs/hub_fastdllm_train}"
# shellcheck disable=SC1091
source "${HUB_TRAIN_VENV}/bin/activate"
export PATH="${HUB_TRAIN_VENV}/bin:${PATH}"

FASTDLLM_ROOT="${FASTDLLM_ROOT:-${REPO_ROOT}/third_party/Fast-dLLM/v2}"
export PYTHONPATH="${FASTDLLM_ROOT}/../third_party:${PYTHONPATH:-}"
cd "${FASTDLLM_ROOT}"

export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
# LMFlow defaults report_to=wandb; prior fails were ModuleNotFoundError: wandb
# (empty namespace stub in venv). Force no reporters — train must not depend on wandb.
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export REPORT_TO="${REPORT_TO:-none}"

WORK="${HUB_C2_ROOT:-${SCRATCH}/hub_fastdllm_c2}"
MODEL_PATH="${MODEL_PATH:-${WORK}/init_qwen15b}"
DATASET_PATH="${DATASET_PATH:-${WORK}/data/c2_mix}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK}/output/c2_qwen_init_${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/slurm_logs"

# Resume: set RESUME_FROM_CHECKPOINT to a checkpoint-* path, or "auto" to pick
# the newest under OUTPUT_DIR. Do NOT use --overwrite_output_dir when resuming
# (LMFlow auto-detects last_checkpoint only if overwrite is off).
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
if [[ "${RESUME_FROM_CHECKPOINT}" == "auto" ]]; then
  RESUME_FROM_CHECKPOINT="$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -z "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "FATAL: RESUME_FROM_CHECKPOINT=auto but no checkpoint-* under ${OUTPUT_DIR}" >&2
    exit 1
  fi
fi
_OVERWRITE_FLAG="--overwrite_output_dir"
_RESUME_FLAG=""
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  [[ -d "${RESUME_FROM_CHECKPOINT}" ]] \
    || { echo "Missing resume ckpt: ${RESUME_FROM_CHECKPOINT}" >&2; exit 1; }
  _OVERWRITE_FLAG=""
  _RESUME_FLAG="--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
  echo "  RESUME from ${RESUME_FROM_CHECKPOINT}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH=${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_PATH}" ]] || ! compgen -G "${DATASET_PATH}/train_*.json" >/dev/null; then
  echo "Missing C2 LMFlow data under ${DATASET_PATH}" >&2
  exit 1
fi

NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-8}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))
# Resolve MASTER to an IP — hostname/interconnect aliases flake rdzv on Jupiter
# (RendezvousConnectionError / mixed jpbo-*-interconnect vs r23-nod* names).
_MASTER_HOST="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
export MASTER_ADDR="${MASTER_ADDR:-$(getent ahostsv4 "${_MASTER_HOST}" | awk '{print $1; exit}')}"
export MASTER_ADDR="${MASTER_ADDR:-${_MASTER_HOST}}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + (SLURM_JOB_ID % 1000)))}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Give c10d store longer to form before SIGTERM storms.
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export TORCHELASTIC_ERROR_FILE="${TORCHELASTIC_ERROR_FILE:-${OUTPUT_DIR}/torchelastic_error.json}"
# Fast-dLLM modeling uses @torch.compile(flex_attention). Inductor needs
# Triton; aarch64 Jupiter has no wheel → step-0 ModuleNotFoundError: triton.
# Belt-and-suspenders with _maybe_torch_compile in modeling.py.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

# Match C2 paper budget: 6000 steps × GBS 256 × length 2048.
MAX_STEPS="${MAX_STEPS:-6000}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
# GBS = nodes * gpus * micro * accum
GRAD_ACCUM="${GRAD_ACCUM:-$((256 / NUM_PROCESSES))}"
if [[ "${GRAD_ACCUM}" -lt 1 ]]; then
  GRAD_ACCUM=1
fi
LR="${LR:-2e-5}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
DS_CONFIG="${DS_CONFIG:-configs/ds_config_zero2_no_offload.json}"

echo "=== hub_fastdllm_c2_train.sbatch ==="
echo "  python=$(command -v python)"
echo "  model=${MODEL_PATH}"
echo "  data=${DATASET_PATH}"
echo "  out=${OUTPUT_DIR}"
echo "  parallel=${NUM_NODES}x${GPUS_PER_NODE} accum=${GRAD_ACCUM} steps=${MAX_STEPS}"
echo "  master=${MASTER_ADDR}:${MASTER_PORT}"
python -c 'import lmflow, deepspeed, transformers; print("lmflow", lmflow.__file__); print("ds", deepspeed.__version__); print("tf", transformers.__version__)'
# Preflight: AutoPipeline must import without wandb (report_to=none path).
python -c 'from lmflow.pipeline.auto_pipeline import AutoPipeline; print("AutoPipeline ok")'

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_srun_retry.bash"
srun_warmup

# Multi-node DeepSpeed via torchrun (HF Trainer --deepspeed).
srun_retry --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
  --cpu-bind=none \
  bash -c "
set -euo pipefail
export LOCAL_RANK=0
torchrun \
  --nnodes=${NUM_NODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --node_rank=\${SLURM_NODEID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=${SLURM_JOB_ID} \
  train_scripts/finetune.py \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code 1 \
    --dataset_path ${DATASET_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --conversation_template fast_dllm_v2 \
    --num_train_epochs 1 \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LR} \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 0 \
    --pad_mask_token 1 \
    --block_size ${BLOCK_SIZE} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --deepspeed ${DS_CONFIG} \
    --bf16 \
    --run_name hub_c2_qwen_init_${SLURM_JOB_ID} \
    --validation_split_percentage 0 \
    --logging_steps 10 \
    --do_train \
    --report_to ${REPORT_TO} \
    ${_OVERWRITE_FLAG} \
    ${_RESUME_FLAG} \
    --ddp_timeout 72000 \
    --save_steps ${SAVE_STEPS} \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 16 \
    --save_total_limit 3 \
    --gradient_checkpointing 1
"

# Point a stable symlink at the finished run for eval submit.
# Hub eval needs a HF tree with custom code (modeling.py). Trainer.save_model
# writes weights+config; copy remote-code / tokenizer files from init if absent.
_copy_hub_code() {
  local src="$1" dst="$2"
  local f
  for f in modeling.py configuration.py generation_config.json \
           tokenizer.json tokenizer_config.json special_tokens_map.json \
           added_tokens.json merges.txt vocab.json chat_template.jinja; do
    if [[ -f "${src}/${f}" && ! -f "${dst}/${f}" ]]; then
      cp -a "${src}/${f}" "${dst}/${f}"
    fi
  done
  if [[ -d "${src}/assets" && ! -d "${dst}/assets" ]]; then
    cp -a "${src}/assets" "${dst}/assets"
  fi
}
_copy_hub_code "${MODEL_PATH}" "${OUTPUT_DIR}"
# If root has no weights yet, point eval at the newest checkpoint-* and copy code there.
if ! compgen -G "${OUTPUT_DIR}/model.safetensors" >/dev/null \
   && ! compgen -G "${OUTPUT_DIR}/pytorch_model*.bin" >/dev/null \
   && ! compgen -G "${OUTPUT_DIR}/*.safetensors" >/dev/null; then
  _ckpt="$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${_ckpt}" ]]; then
    echo "WARNING: no root weights; eval target → ${_ckpt}"
    _copy_hub_code "${MODEL_PATH}" "${_ckpt}"
    ln -sfn "${_ckpt}" "${WORK}/output/latest"
  else
    echo "FATAL: train finished but no model weights under ${OUTPUT_DIR}" >&2
    exit 1
  fi
else
  ln -sfn "${OUTPUT_DIR}" "${WORK}/output/latest"
fi
echo "TRAIN_DONE output=${OUTPUT_DIR} latest=$(readlink -f "${WORK}/output/latest")"
