#!/usr/bin/env bash
# One-time sync: login-only /p/project1/... -> compute-visible /e/project1/...
#
# Jupiter compute nodes cannot see /p/project1. SLURM jobs need the repo,
# venv, and data_cache under /e/project1/scifi/elsayed3/JEDi.
#
# Usage (on login node):
#   bash slurm_scripts/block_diffusion/setup_compute_workspace.sh
#
# Re-run safely after code changes (rsync incremental).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jupiter_paths.sh
source "${SCRIPT_DIR}/jupiter_paths.sh"

SRC="$(resolve_jedi_repo_root)"
if [[ "${SRC}" == "${JEDI_COMPUTE_ROOT}" ]] && [[ -d "${JEDI_LOGIN_ROOT}" ]]; then
  SRC="${JEDI_LOGIN_ROOT}"
fi
DST="${JEDI_COMPUTE_ROOT}"

echo "=========================================="
echo "Jupiter compute workspace sync"
echo "  FROM: ${SRC}  (login /p mount)"
echo "  TO:   ${DST}  (compute /e mount — scifi convention)"
echo ""
echo "Other scifi projects (amir, fourel1, Kalyan) keep code under"
echo "/e/project1/scifi/<user>/ and never reference /p/ in batch jobs."
echo "=========================================="

mkdir -p "$(dirname "${DST}")"

RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude 'outputs/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.ipynb_checkpoints/'
)

rsync -a --info=stats2,progress2 "${RSYNC_EXCLUDES[@]}" "${SRC}/" "${DST}/"

echo "=========================================="
echo "Sync complete."
echo "  Compute repo: ${DST}"
echo "  Submit jobs from either path; scripts use ${DST} on compute."
echo "=========================================="
