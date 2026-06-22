#!/usr/bin/env bash
# BlockDiffusion OWT finetune from BD3-LM MDLM pretrain (Baseline B).
#
# Prerequisite (login node, once):
#   bash slurm_scripts/block_diffusion/download_mdlm_pretrain.sh
#
# Submit:
#   sbatch slurm_scripts/block_diffusion/owt_from_pretrain.sh
#   MODE=uniform sbatch slurm_scripts/block_diffusion/owt_from_pretrain.sh
#
# Or:
#   FROM_PRETRAINED=1 sbatch slurm_scripts/block_diffusion/owt.sh

export FROM_PRETRAINED="${FROM_PRETRAINED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/owt.sh"
