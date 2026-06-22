#!/usr/bin/env bash
# Login-node smoke test before submitting BlockDiffusion SLURM jobs.
#
# Catches common failures (venv, triton/flex, data caches, checkpoints, imports)
# without allocating GPU time on Booster.
#
# Usage:
#   bash slurm_scripts/block_diffusion/smoke_test_login.sh
#   bash slurm_scripts/block_diffusion/smoke_test_login.sh owt
#   PROFILE=owt-pretrain bash slurm_scripts/block_diffusion/smoke_test_login.sh
#
# Optional GPU flex forward (5 min, 1 GPU) after login checks pass:
#   bash slurm_scripts/block_diffusion/smoke_test_gpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jupiter_paths.sh
source "${SCRIPT_DIR}/jupiter_paths.sh"

REPO_ROOT="$(resolve_jedi_repo_root)"
# On login, prefer /p checkout (where you edit); compute_sync still checks /e.
if [[ -z "${SLURM_JOB_ID:-}" ]] && [[ -d "${JEDI_LOGIN_ROOT}/.venv" ]]; then
  REPO_ROOT="${JEDI_LOGIN_ROOT}"
fi
PROFILE="${1:-${PROFILE:-all}}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONSTARTUP="${REPO_ROOT}/scripts/triton_sitecustomize.py"
export REPO_ROOT
export JEDI_COMPUTE_ROOT

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: venv missing at ${REPO_ROOT}/.venv" >&2
  echo "Run: bash slurm_scripts/block_diffusion/setup_cuda_venv.sh" >&2
  exit 1
fi

echo "=========================================="
echo "BlockDiffusion login smoke test"
echo "  REPO:     ${REPO_ROOT}"
echo "  COMPUTE:  ${JEDI_COMPUTE_ROOT}"
echo "  PROFILE:  ${PROFILE}"
echo "=========================================="

"${PYTHON}" "${REPO_ROOT}/scripts/smoke_block_diffusion_login.py" \
  --profile "${PROFILE}" \
  --repo-root "${REPO_ROOT}" \
  --compute-root "${JEDI_COMPUTE_ROOT}"
PY_EXIT=$?

if [[ "${PY_EXIT}" -ne 0 ]]; then
  exit "${PY_EXIT}"
fi

run_dry_run() {
  local label="$1"
  shift
  echo ""
  echo "--- DRY_RUN: ${label} ---"
  DRY_RUN=1 env "$@" bash "${SCRIPT_DIR}/train_core.sh"
}

case "${PROFILE}" in
  tiny)
    run_dry_run "tiny masked" \
      MODE=masked DATA=tiny_shakespeare ATTN_BACKEND=sdpa SEQ_LEN=128 \
      GLOBAL_BATCH=64 MAX_STEPS=2 NUM_WORKERS=0
    ;;
  owt)
    run_dry_run "OWT scratch masked" \
      MODE=masked DATA=openwebtext-split ATTN_BACKEND=flex SEQ_LEN=1024 \
      GLOBAL_BATCH=512 MAX_STEPS=2 NUM_WORKERS=0
    ;;
  owt-pretrain)
    run_dry_run "OWT MDLM pretrain" \
      MODE=masked DATA=openwebtext-split FROM_PRETRAINED=1 ATTN_BACKEND=flex \
      SEQ_LEN=1024 GLOBAL_BATCH=512 MAX_STEPS=2 NUM_WORKERS=0
    ;;
  owt-ar)
    run_dry_run "OWT AR pretrain" \
      MODE=masked DATA=openwebtext-split FROM_AR_PRETRAINED=1 ATTN_BACKEND=flex \
      SEQ_LEN=1024 GLOBAL_BATCH=512 MAX_STEPS=2 NUM_WORKERS=0
    ;;
  all)
    run_dry_run "tiny masked" \
      MODE=masked DATA=tiny_shakespeare ATTN_BACKEND=sdpa SEQ_LEN=128 \
      GLOBAL_BATCH=64 MAX_STEPS=2 NUM_WORKERS=0
    run_dry_run "OWT scratch masked" \
      MODE=masked DATA=openwebtext-split ATTN_BACKEND=flex SEQ_LEN=1024 \
      GLOBAL_BATCH=512 MAX_STEPS=2 NUM_WORKERS=0
    run_dry_run "OWT MDLM pretrain" \
      MODE=masked DATA=openwebtext-split FROM_PRETRAINED=1 ATTN_BACKEND=flex \
      SEQ_LEN=1024 GLOBAL_BATCH=512 MAX_STEPS=2 NUM_WORKERS=0
    run_dry_run "OWT AR pretrain" \
      MODE=masked DATA=openwebtext-split FROM_AR_PRETRAINED=1 ATTN_BACKEND=flex \
      SEQ_LEN=1024 GLOBAL_BATCH=512 MAX_STEPS=2 NUM_WORKERS=0
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

echo ""
echo "Login smoke test complete."
