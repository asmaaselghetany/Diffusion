#!/usr/bin/env bash
#SBATCH --job-name=ts_jepa_eval
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err

set -euo pipefail

export MODEL=latent_jepa
export REPO_ROOT="${REPO_ROOT:-/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi}"
export NUM_SAMPLES="${NUM_SAMPLES:-64}"
export NUM_STEPS="${NUM_STEPS:-64}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
export EVAL_MODEL="${EVAL_MODEL:-gpt2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export SEQ_LEN="${SEQ_LEN:-128}"

CORE_DIR="${REPO_ROOT}/slurm_scripts/shakespeare_baselines"
"${CORE_DIR}/eval_core.sh"
