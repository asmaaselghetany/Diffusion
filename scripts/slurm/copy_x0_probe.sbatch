#!/usr/bin/env bash
#SBATCH --job-name=copy_x0_probe
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --chdir=/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
UNI_D2_ROOT="${UNI_D2_ROOT:-${REPO_ROOT}}"
# shellcheck disable=SC1091
source "${UNI_D2_ROOT}/.venv/bin/activate"
export PYTHONPATH="${UNI_D2_ROOT}/src"
module load CUDA/12.6.0 2>/dev/null || true

CKPT="${1:?Usage: sbatch copy_x0_probe.sbatch <ckpt> <tag>}"
TAG="${2:-probe}"
OUT="${UNI_D2_ROOT}/outputs/block_qwen/copy_x0_${TAG}.json"

python tools/run_copy_x0_probe.py \
  --checkpoint "${CKPT}" \
  --device cuda \
  --seq-len 512 \
  --t 0.05 \
  --real-batch \
  --output "${OUT}"

echo "Wrote ${OUT}"
