#!/usr/bin/env bash
#SBATCH --job-name=bqwen-genppl-hyg
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Unifusion-style GenPPL hygiene: dual modes, NFE sweep {8,16,32,64},
# collapse panel; optional multiseed (RUN_MULTISEED=1).
#
#   CKPT=... MODE=all sbatch --export=ALL,CKPT,MODE,NUM_STEPS_LIST,RUN_MULTISEED \
#     scripts/slurm/gen_ppl_hygiene.sbatch

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

CKPT="${CKPT:?Set CKPT=/path/to.ckpt}"
MODE="${MODE:-all}"
export CKPT MODE
export NUM_STEPS_LIST="${NUM_STEPS_LIST:-8 16 32 64}"
export NUM_SAMPLES="${NUM_SAMPLES:-64}"
export SAMPLE_MODE="${SAMPLE_MODE:-conversion_free}"
export DECODE_PROFILE="${DECODE_PROFILE:-hierarchical_ss}"
export RUN_MULTISEED="${RUN_MULTISEED:-0}"
export SEEDS="${SEEDS:-1 2 3}"

bash scripts/run_gen_ppl_hygiene.sh
