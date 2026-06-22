#!/usr/bin/env bash
# Download kuleshov-group/bd3lm-owt-block_size1024-pretrain for offline OWT finetuning.
#
# Run once on a Jupiter login node (has internet). Writes:
#   data_cache/checkpoints/bd3lm_owt_block1024_pretrain.ckpt
#
# Then submit with FROM_PRETRAINED=1:
#   FROM_PRETRAINED=1 sbatch slurm_scripts/block_diffusion/owt.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_ROOT}/data_cache}"

cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" scripts/download_bd3lm_pretrain.py --cache-dir "${DATA_CACHE_DIR}"

echo ""
echo "Sync to compute path if needed:"
echo "  bash slurm_scripts/block_diffusion/setup_compute_workspace.sh"
