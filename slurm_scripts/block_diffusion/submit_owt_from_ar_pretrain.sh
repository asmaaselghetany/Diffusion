#!/usr/bin/env bash
# Submit masked + uniform OWT jobs from AR pretrain (AR → block diffusion).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jupiter_paths.sh
source "${SCRIPT_DIR}/jupiter_paths.sh"
REPO_ROOT="$(resolve_jedi_repo_root)"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_ROOT}/data_cache}"
PRETRAIN_CKPT="${DATA_CACHE_DIR}/checkpoints/ar_noeos_owt.ckpt"
EOS_FALSE_CACHE="${DATA_CACHE_DIR}/openwebtext-train_train_bs1024_wrapped_eosFalse_specialFalse.dat"

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "ERROR: Missing ${PRETRAIN_CKPT}" >&2
  echo "Run: bash slurm_scripts/block_diffusion/download_ar_pretrain.sh" >&2
  exit 1
fi

if [[ ! -d "${EOS_FALSE_CACHE}" ]] && [[ ! -f "${EOS_FALSE_CACHE}" ]]; then
  echo "ERROR: Missing pretrain-aligned OWT cache:" >&2
  echo "  ${EOS_FALSE_CACHE}" >&2
  echo "Run: PRETRAIN_DATA=1 SPECS=\"openwebtext-split:1024\" bash slurm_scripts/block_diffusion/prep_data.sh" >&2
  exit 1
fi

for mode in masked uniform; do
  echo "Submitting MODE=${mode} FROM_AR_PRETRAINED=1 ..."
  MODE="${mode}" FROM_AR_PRETRAINED=1 DATA_CACHE_DIR="${DATA_CACHE_DIR}" \
    sbatch "${SCRIPT_DIR}/owt_from_ar_pretrain.sh"
done
