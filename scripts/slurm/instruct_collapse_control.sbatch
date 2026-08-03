#!/usr/bin/env bash
#SBATCH --job-name=bqwen-ctrl
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --chdir=/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2
#
# Instruct collapse control (independent of DepBench / finetuned ckpts):
#   sbatch scripts/slurm/instruct_collapse_control.sbatch

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

OUT="outputs/block_qwen/controls/instruct_collapse"
mkdir -p "${OUT}" slurm_logs

srun python -u tools/run_instruct_collapse_control.py \
  --out-dir "${OUT}" \
  --num-samples 4 \
  --ar-max-new-tokens 128 \
  --length 256 \
  --block-size 32 \
  --steps 32 \
  --algos block_masked,block_uniform

echo "Done. See ${OUT}/control_report.json"
