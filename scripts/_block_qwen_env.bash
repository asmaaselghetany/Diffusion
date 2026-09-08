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
activate_jupiter_modules
if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDA_HOME:-}${CUDA_ROOT:-}" ]]; then
  module load CUDA 2>/dev/null || true
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# CRITICAL: Slurm propagates the submitter's env. A polluted PYTHONPATH that
# prefixes projects/depbench/.deps-fastdllm (transformers 4.53) breaks the
# Qwen block-attn monkeypatch — 4.53 Qwen2Model has no _update_causal_mask
# (Track 1 micros 137328/137329). Original paper arms used clean venv
# 4.45. Always reset to uni-d2 src only after activate.
export PYTHONPATH="${REPO_ROOT}/src"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# WandB: always load key from workspace secrets if missing, then force online.
# Do NOT honor a stale WANDB_MODE=offline inherited from the submit shell
# (Slurm propagates the submitter env — that bit us after offline rebuilds).
_WANDB_KEY_FILE="${ASMAA_WORKSPACE}/.secrets/wandb_api_key"
if [[ -z "${WANDB_API_KEY:-}" && -f "${_WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(<"${_WANDB_KEY_FILE}")"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=online
else
  echo "WARNING: WANDB_API_KEY unset; logging offline" >&2
  export WANDB_MODE=offline
fi
# Prevent tqdm/progress-bar spam from flooding WandB filestream (causes 429, no charts).
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
# Don't let wandb.init hang forever and desync DDP ranks.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-60}"
# Prefer scratch cache from env.sh; do not clobber a caller-set path.
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DISCRETE_DIFFUSION_SCRATCH_DIR:-${SCRATCH:-${ASMAA_WORKSPACE}}/.cache/discrete_diffusion}"
# Local per-job temp avoids pymp-* contention on shared project FS (Lustre/NFS).
export TMPDIR="/tmp/block_qwen_${SLURM_JOB_ID:-local}"
mkdir -p "${TMPDIR}"

# Checkpoint safety: refuse to start if the project FS cannot absorb a ~23G write.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_block_qwen_ckpt.bash"
_block_qwen_require_disk "${REPO_ROOT}" || exit 1

mkdir -p slurm_logs outputs
