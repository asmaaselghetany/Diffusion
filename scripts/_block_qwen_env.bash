# Shared SLURM runtime setup for Qwen block jobs (source from scripts/*.sbatch).
set -euo pipefail

# Jupiter compute sees /e/project1; legacy /fast path is login-only elsewhere.
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
export ASMAA_WORKSPACE="${WORKSPACE}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

# Prefer Jupiter Stages/CUDA stack; fall back to a bare CUDA module if present.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/jupiter_paths.sh"
# Pin TMPDIR to *this* Slurm job before module load (mktemp). Inherited
# TMPDIR from a parent train job points at a missing /tmp/block_qwen_<old>.
export TMPDIR="/tmp/block_qwen_${SLURM_JOB_ID:-local}"
mkdir -p "${TMPDIR}"
activate_jupiter_modules
if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDA_HOME:-}${CUDA_ROOT:-}" ]]; then
  module load CUDA 2>/dev/null || true
fi
# Prefer activate; fall back to PATH-only if bin/activate was wiped (venv restore).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -x .venv/bin/python ]]; then
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
  export PYTHONNOUSERSITE=1
else
  echo "FATAL: missing ${REPO_ROOT}/.venv/bin/python (and no activate)" >&2
  exit 1
fi

# CRITICAL: Slurm propagates the submitter's env. A polluted PYTHONPATH that
# prefixes projects/depbench/.deps-fastdllm (transformers 4.53) breaks the
# Qwen block-attn monkeypatch — 4.53 Qwen2Model has no _update_causal_mask
# (Track 1 micros 137328/137329). Original paper arms used clean venv
# 4.45. Always reset to uni-d2 src only after activate.
export PYTHONPATH="${REPO_ROOT}/src"

# Refuse stub / drifted Unif helpers before any train or eval work.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_assert_utils_hash.bash"
assert_forward_process_utils_hash || exit $?

# Booster compute has no route to huggingface.co. Prefer local HF_HOME cache
# (env.sh → ${SCRATCH}/hf_cache). Without these, transformers≥4.4x issues
# HEAD requests even when snapshots exist and hangs/fails the train start.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# WandB: load key from env / WANDB_API_KEY_FILE / ~/.config, then workspace
# secrets (cluster). Compute nodes on Jupiter have no route to api.wandb.ai —
# default OFFLINE under Slurm (sync later with upload_wandb_panels). Online
# only when WANDB_FORCE_ONLINE=1 (login-node smoke) or non-Slurm + key set.
# Soft xfer 2012251/2012253 failed: online wandb.init hung → DDP NCCL timeout.
# Hygiene afterok used --export=CKPT,... without HOME; set -u then crashed here.
: "${HOME:=${ASMAA_HOME:-/e/home/jusers/elsayed3/jupiter}}"
export HOME
_WANDB_KEY_FILE_HOME="${HOME}/.config/wandb/api_key"
_WANDB_KEY_FILE_WS="${ASMAA_WORKSPACE}/.secrets/wandb_api_key"
if [[ -z "${WANDB_API_KEY:-}" && -n "${WANDB_API_KEY_FILE:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
elif [[ -z "${WANDB_API_KEY:-}" && -f "${_WANDB_KEY_FILE_HOME}" ]]; then
  export WANDB_API_KEY="$(<"${_WANDB_KEY_FILE_HOME}")"
elif [[ -z "${WANDB_API_KEY:-}" && -f "${_WANDB_KEY_FILE_WS}" ]]; then
  export WANDB_API_KEY="$(<"${_WANDB_KEY_FILE_WS}")"
fi
# Team entity hosting block_qwen* projects (username aselghetany → entity aselghetany-nu).
export WANDB_ENTITY="${WANDB_ENTITY:-aselghetany-nu}"
if [[ "${WANDB_FORCE_OFFLINE:-0}" == "1" ]]; then
  export WANDB_MODE=offline
elif [[ "${WANDB_FORCE_ONLINE:-0}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=online
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Booster compute: no egress to wandb. Online desyncs rank0 vs NCCL.
  export WANDB_MODE=offline
  export WANDB_FORCE_OFFLINE=1
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=online
else
  echo "WARNING: WANDB_API_KEY unset; logging offline" >&2
  export WANDB_MODE=offline
fi
# Console off during DDP train: tqdm spam caused WandB 429s / missing charts.
# Slurm stdout is mirrored into the run Logs tab post-hoc by
# scripts/upload_wandb_panels.py (LOGS_ONLY=1 / eval attach).
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
# Don't let wandb.init hang forever and desync DDP ranks.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-60}"
# Prefer scratch cache from env.sh; do not clobber a caller-set path.
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DISCRETE_DIFFUSION_SCRATCH_DIR:-${SCRATCH:-${ASMAA_WORKSPACE}}/.cache/discrete_diffusion}"

# Checkpoint safety: refuse to start if the project FS cannot absorb a ~23G write.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_block_qwen_ckpt.bash"
_block_qwen_require_disk "${REPO_ROOT}" || exit 1

mkdir -p slurm_logs outputs
