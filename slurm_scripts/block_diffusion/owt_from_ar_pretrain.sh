#!/usr/bin/env bash
# BlockDiffusion OWT finetune from BD3-LM AR baseline (AR → block diffusion).
#
# Prerequisite (login node, once):
#   bash slurm_scripts/block_diffusion/download_ar_pretrain.sh
#   PRETRAIN_DATA=1 SPECS="openwebtext-split:1024" bash slurm_scripts/block_diffusion/prep_data.sh
#
# Submit:
#   sbatch slurm_scripts/block_diffusion/owt_from_ar_pretrain.sh
#   MODE=uniform sbatch slurm_scripts/block_diffusion/owt_from_ar_pretrain.sh

export FROM_AR_PRETRAINED="${FROM_AR_PRETRAINED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/owt.sh"
