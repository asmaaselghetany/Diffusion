#!/usr/bin/env bash
#SBATCH --job-name=bqwen-block-masked
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --chdir=/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2
#
# Pipeline 2 (pure block diffusion / BlockGen style): scratch Qwen arch →
# block_masked SFT. No AR pretrained weights. Hooks OFF.
# Pair with block_uniform.sbatch.
#
# Submit: sbatch scripts/slurm/block_masked.sbatch
#   or:   ./scripts/submit_block.sh both

set -euo pipefail

export LINE=block
export NUM_GPUS=2
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2/outputs/block_qwen/block_masked_${SLURM_JOB_ID}"
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash
# shellcheck disable=SC1091
source scripts/_block_qwen_launch.bash masked
