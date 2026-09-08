#!/usr/bin/env bash
#SBATCH --job-name=bqwen-masked
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
# Legacy alias → Pipeline 1 AR→block masked.
# Prefer: sbatch scripts/slurm/ar2block_masked.sbatch
#
# When RUN_ROOT is already set (resume of masked_<jobid>), keep that path.
# Otherwise write to ar2block_masked_<jobid>.

set -euo pipefail

export LINE=ar2block
export NUM_NODES=1
export GPUS_PER_NODE=2
export NUM_GPUS=2
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2/outputs/block_qwen/ar2block_masked_${SLURM_JOB_ID}"
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash
# shellcheck disable=SC1091
source scripts/_block_qwen_launch.bash masked
