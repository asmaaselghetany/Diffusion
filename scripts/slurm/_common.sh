# Shared SLURM env for block_qwen on Jupiter Booster.
# Source from *.sbatch:  source scripts/slurm/_common.sh
#
# Sets up /e/ paths, modules, venv, offline HF caches, and DDP networking.
set -uo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../jupiter_paths.sh
source "${_SCRIPT_DIR}/../jupiter_paths.sh"

REPO_ROOT="${REPO_ROOT:-$(resolve_diffusion_repo_root)}"
export REPO_ROOT ASMAA_WORKSPACE="${ASMAA_WORKSPACE:-${DIFFUSION_WORKSPACE}}"

# Translate SLURM_SUBMIT_DIR if captured under /p/ on login node.
_PROJ="${SCIFI_PROJECT}"
_SCR="${SCIFI_SCRATCH}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ ! -d "${SLURM_SUBMIT_DIR}" ]]; then
  _ALT="$(translate_scifi_path "${SLURM_SUBMIT_DIR}")"
  if [[ -d "${_ALT}" ]]; then
    echo "[common] path-translate ${SLURM_SUBMIT_DIR} -> ${_ALT}"
    SLURM_SUBMIT_DIR="${_ALT}"
    export SLURM_SUBMIT_DIR
  fi
fi

cd "${REPO_ROOT}"

if command -v jutil >/dev/null 2>&1; then
  jutil env activate -p "${JUWELS_PROJECT:-scifi}" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "${ASMAA_WORKSPACE}/env.sh" 2>/dev/null || true
export SCRATCH="${SCRATCH:-${SCIFI_SCRATCH}/elsayed3}"
export REPO_ROOT

activate_jupiter_modules

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  require_diffusion_repo "${REPO_ROOT}" || exit 99
fi

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
hash -r

export PYTHONPATH="${REPO_ROOT}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-72}}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DISCRETE_DIFFUSION_SCRATCH_DIR:-${SCRATCH}/.cache/discrete_diffusion}"
export DATA_CACHE="${DATA_CACHE:-${DISCRETE_DIFFUSION_SCRATCH_DIR}/block_qwen_sft_nemotron}"
# Drop partial dataset caches from interrupted jobs (breaks HF offline load).
find "${DATA_CACHE}" -name '*.incomplete' -type d -exec rm -rf {} + 2>/dev/null || true

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  _first_host="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
  # Jupiter: bare hostname resolves; the JUWELS "hostname+i" IB alias does not.
  # Single-node multi-GPU: one Slurm task per GPU (external DDP ranks).
  if [[ "${SLURM_NNODES:-1}" -le 1 ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  else
    export MASTER_ADDR="${MASTER_ADDR:-${_first_host}}"
  fi
  export MASTER_PORT="${MASTER_PORT:-29500}"
  echo "[common] ddp master=${MASTER_ADDR}:${MASTER_PORT} nnodes=${SLURM_NNODES:-1}"
fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"
export SRUN_CPUS_PER_TASK="${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-72}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export TMPDIR="/tmp/block_qwen_${SLURM_JOB_ID:-local}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${DISCRETE_DIFFUSION_SCRATCH_DIR}" slurm_logs outputs

echo "[common] node=$(hostname) job=${SLURM_JOB_ID:-local} repo=${REPO_ROOT}"
echo "[common] scratch=${SCRATCH} HF_HOME=${HF_HOME} offline=${HF_HUB_OFFLINE}"
nvidia-smi -L 2>&1 | sed 's/^/[common]   /' || true

# JSC requires --account on every sbatch, even with --export=ALL.
sbatch_with_account() {
  local account_arg=()
  if [[ -n "${JUWELS_ACCOUNT:-}" ]]; then
    account_arg=(--account="${JUWELS_ACCOUNT}")
  else
    account_arg=(--account=scifi)
  fi
  sbatch "${account_arg[@]}" "$@"
}
