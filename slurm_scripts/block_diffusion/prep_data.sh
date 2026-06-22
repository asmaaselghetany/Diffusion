#!/usr/bin/env bash
# One-time data prep on a login node (internet required).
#
# Downloads raw data, caches the GPT-2 tokenizer, and writes tokenized
# .dat shards for offline BlockDiffusion SLURM jobs.
#
# Usage:
#   bash slurm_scripts/block_diffusion/prep_data.sh
#
# Prep only OWT (skip tiny Shakespeare if already cached):
#   SPECS="openwebtext-split:1024" bash slurm_scripts/block_diffusion/prep_data.sh
#
# OWT for MDLM pretrain finetune (no EOS/special tokens — matches FROM_PRETRAINED=1):
#   PRETRAIN_DATA=1 SPECS="openwebtext-split:1024" bash slurm_scripts/block_diffusion/prep_data.sh
#
# Custom cache location:
#   DATA_CACHE_DIR=/path/to/cache bash slurm_scripts/block_diffusion/prep_data.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-}"
SCRIPT_DIR_PREP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jupiter_paths.sh
source "${SCRIPT_DIR_PREP}/jupiter_paths.sh"
REPO_ROOT="$(resolve_jedi_repo_root)"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_ROOT}/data_cache}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Match seq lengths used by the BlockDiffusion SLURM launchers:
#   tiny_shakespeare.sh -> SEQ_LEN=128
#   owt.sh              -> SEQ_LEN=1024
DEFAULT_SPECS="tiny_shakespeare:128 openwebtext-split:1024"
read -r -a SPEC_LIST <<< "${SPECS:-${DEFAULT_SPECS}}"

cd "${REPO_ROOT}"

set +u
source ~/.bashrc 2>/dev/null || true
set -u

PYTHON_CMD=(python)
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
else
  source "${REPO_ROOT}/.venv/bin/activate" 2>/dev/null || true
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${DATA_CACHE_DIR}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${DATA_CACHE_DIR}/hf_hub"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${DATA_CACHE_DIR}"

PREP_CMD=("${PYTHON_CMD[@]}" "${REPO_ROOT}/scripts/prep_data_cache.py")
PREP_CMD+=(--cache-dir "${DATA_CACHE_DIR}")
PREP_CMD+=(--num-workers "${NUM_WORKERS}")
if [[ -n "${PRETRAIN_DATA:-}" ]] && [[ "${PRETRAIN_DATA}" != "0" ]]; then
  PREP_CMD+=(--pretrain-align)
fi
for spec in "${SPEC_LIST[@]}"; do
  PREP_CMD+=(--spec "${spec}")
done

echo "=========================================="
echo "BlockDiffusion data prep (login node)"
echo "  REPO_ROOT:      ${REPO_ROOT}"
echo "  DATA_CACHE_DIR: ${DATA_CACHE_DIR}"
echo "  SPECS:          ${SPEC_LIST[*]}"
echo "  NUM_WORKERS:    ${NUM_WORKERS}"
echo "=========================================="

"${PREP_CMD[@]}"

echo "=========================================="
echo "Done. You can now submit SLURM jobs with HF_HUB_OFFLINE=1."
echo "  sbatch slurm_scripts/block_diffusion/tiny_shakespeare.sh"
echo "  sbatch slurm_scripts/block_diffusion/owt.sh"
echo "=========================================="
