#!/usr/bin/env bash
#SBATCH --job-name=bqwen-ar2block-masked
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
# Pipeline 1 (AR→block / Fast-dLLM style): pretrained Qwen → block_masked SFT.
# Hooks OFF. Pair with ar2block_uniform.sbatch.
#
# Submit: sbatch scripts/slurm/ar2block_masked.sbatch
#   or:   ./scripts/submit_ar2block.sh both

set -euo pipefail

export LINE=ar2block
export NUM_GPUS=2
# Preserve RUN_ROOT when resume_block_qwen.sh pins an existing run dir.
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2/outputs/block_qwen/ar2block_masked_${SLURM_JOB_ID}"
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash
# shellcheck disable=SC1091
source scripts/_block_qwen_launch.bash masked
