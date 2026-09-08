#!/usr/bin/env bash
#SBATCH --job-name=bqwen-blockgen-masked
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
# Legacy alias for Pipeline 2 scratch masked.
# Prefer: scripts/slurm/block_masked.sbatch (LINE=block, dirs block_masked_*).
# Kept so resume of blockgen_masked_<jobid> still works.

set -euo pipefail

export LINE=blockgen
export NUM_NODES=1
export GPUS_PER_NODE=2
export NUM_GPUS=2
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2/outputs/block_qwen/blockgen_masked_${SLURM_JOB_ID}"
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash
# shellcheck disable=SC1091
source scripts/_block_qwen_launch.bash masked
