#!/usr/bin/env bash
# Download kuleshov-group/ar-noeos-owt for offline AR → block-diffusion finetuning.
#
# Run once on a Jupiter login node (has internet). Writes:
#   data_cache/checkpoints/ar_noeos_owt.ckpt
#
# Then submit with:
#   bash slurm_scripts/block_diffusion/submit_owt_from_ar_pretrain.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_ROOT}/data_cache}"

cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" scripts/download_ar_pretrain.py --cache-dir "${DATA_CACHE_DIR}"

echo ""
echo "Sync to compute path if needed:"
echo "  rsync -a ${DATA_CACHE_DIR}/checkpoints/ar_noeos_owt.ckpt /e/project1/scifi/elsayed3/JEDi/data_cache/checkpoints/"
