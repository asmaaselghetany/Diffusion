#!/usr/bin/env bash
#SBATCH --job-name=bqwen-ar2block-masked
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Pipeline 1 (AR→block / Fast-dLLM style): pretrained Qwen → block_masked SFT.
# Hooks OFF unless HYDRA_OVERRIDES / lever submitter enables them.
# Booster QOS MaxWall=12h: paper 6000-step runs need multi-node (NUM_NODES≥8 default).
#
# Submit: sbatch scripts/slurm/ar2block_masked.sbatch
#   or:   NUM_NODES=8 ./scripts/submit_lever.sh --preset C2_fdllm_full --arm masked

set -euo pipefail

export ASMAA_WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
export LINE=ar2block
export NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_GPUS="${NUM_GPUS:-$((NUM_NODES * GPUS_PER_NODE))}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="${ASMAA_WORKSPACE}/Diffusion-new/outputs/block_qwen/ar2block_masked_${SLURM_JOB_ID}"
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash
# shellcheck disable=SC1091
source scripts/_block_qwen_launch.bash masked
