#!/usr/bin/env bash
#SBATCH --job-name=bqwen-verify
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --open-mode=append
#SBATCH --chdir=/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2

set -euo pipefail

# SLURM copies this script to /var/spool/... — do not resolve paths via BASH_SOURCE.
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

srun bash examples/block_qwen/smoke.sh
