#!/usr/bin/env bash
#SBATCH --job-name=hub-fdllm-lm
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Official Hub Fast-dLLM v2 weights + vendored eval.py (paper Table 1 calibration).
#
#   MODEL_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B THRESHOLD=1 \
#     sbatch scripts/slurm/hub_fastdllm_lm_eval.sbatch

set -euo pipefail

export ASMAA_WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# Do NOT use scripts/_block_qwen_env.bash (activates train venv @ transformers 4.45).
# shellcheck disable=SC1091
source "${ASMAA_WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/jupiter_paths.sh"
activate_jupiter_modules

HUB_VENV="${HUB_VENV:-/e/scratch/scifi/elsayed3/.venvs/hub_fastdllm}"
# shellcheck disable=SC1091
source "${HUB_VENV}/bin/activate"
export PATH="${HUB_VENV}/bin:${PATH}"
export PYTHONPATH=""

export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"
export TOKENIZERS_PARALLELISM=false

export MODEL_PATH="${MODEL_PATH:-/e/scratch/scifi/elsayed3/hf_cache/models/Fast_dLLM_v2_1.5B}"
export THRESHOLD="${THRESHOLD:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export SMALL_BLOCK_SIZE="${SMALL_BLOCK_SIZE:-8}"
export BD_SIZE="${BD_SIZE:-32}"
export USE_BLOCK_CACHE="${USE_BLOCK_CACHE:-False}"
export TASKS="${TASKS:-mmlu,gsm8k,ifeval}"
export OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/hub_fastdllm_eval/$(basename "${MODEL_PATH}")_t${THRESHOLD}}"
mkdir -p "${OUT_DIR}" slurm_logs

export NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-4}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_NODES * GPUS_PER_NODE))}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + (SLURM_JOB_ID % 1000)))}"
# Prefer IPv4 — interconnect DNS aliases previously broke accelerate AF_INET6.
_MASTER_HOST="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
export MASTER_ADDR="${MASTER_ADDR:-$(getent ahostsv4 "${_MASTER_HOST}" | awk '{print $1; exit}')}"
export MASTER_ADDR="${MASTER_ADDR:-${_MASTER_HOST}}"
# Match scripts/slurm/_common.sh — without ib0, accelerate hits AF_INET6
# "Address family not supported" on Jupiter interconnect hostnames.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export LM_EVAL_DIST_TIMEOUT_SEC="${LM_EVAL_DIST_TIMEOUT_SEC:-21600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${LM_EVAL_DIST_TIMEOUT_SEC}}"

# Fail fast if MODEL_PATH is a dangling symlink (train did not publish latest).
if [[ -L "${MODEL_PATH}" && ! -e "${MODEL_PATH}" ]]; then
  echo "FATAL: MODEL_PATH symlink dangling: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "FATAL: MODEL_PATH missing: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "FATAL: no config.json under MODEL_PATH=${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/modeling.py" ]]; then
  echo "FATAL: no modeling.py under MODEL_PATH=${MODEL_PATH} (Hub remote code)" >&2
  exit 1
fi
if ! compgen -G "${MODEL_PATH}/model.safetensors" >/dev/null \
   && ! compgen -G "${MODEL_PATH}/pytorch_model*.bin" >/dev/null \
   && ! compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null; then
  echo "FATAL: no weight files under MODEL_PATH=${MODEL_PATH}" >&2
  exit 1
fi

echo "=== hub_fastdllm_lm_eval.sbatch ==="
echo "  python=$(command -v python)  transformers=$(python -c 'import transformers; print(transformers.__version__)')"
echo "  model=${MODEL_PATH} thr=${THRESHOLD} tasks=${TASKS}"
echo "  out=${OUT_DIR} parallel=${NUM_NODES}x${GPUS_PER_NODE}"
echo "  master=${MASTER_ADDR}:${MASTER_PORT}"

# shellcheck disable=SC1091
source scripts/_srun_retry.bash
srun_warmup

srun_retry --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
  --cpu-bind=none \
  bash examples/block_qwen/hub_fastdllm_lm_eval.sh
