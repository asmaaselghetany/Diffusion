#!/usr/bin/env bash
# Local smoke: dry-run OWT block diffusion from MDLM pretrain.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

PRETRAIN="${REPO_ROOT}/data_cache/checkpoints/bd3lm_owt_block1024_pretrain.ckpt"
if [[ ! -f "${PRETRAIN}" ]]; then
  echo "Run: bash slurm_scripts/block_diffusion/download_mdlm_pretrain.sh"
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MODE="${MODE:-masked}"
export FROM_PRETRAINED=1
export DATA=openwebtext-split
export DATA_CACHE_DIR="${REPO_ROOT}/data_cache"
export OUTPUT_BASE="${REPO_ROOT}/outputs/block_diffusion/owt_smoke_pretrain"
export SEQ_LEN=1024
export BLOCK_SIZE=16
export GLOBAL_BATCH=64
export MAX_STEPS=2
export VAL_CHECK_INTERVAL=100000
export NUM_WORKERS=0
export DRY_RUN=1

bash slurm_scripts/block_diffusion/train_core.sh
