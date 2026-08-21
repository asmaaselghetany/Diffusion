#!/usr/bin/env bash
#SBATCH --job-name=arm_sanity
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

CKPT="${1:?Usage: sbatch arm_sanity.sbatch <ckpt> [tag]}"
TAG="${2:-paper_masked}"
OUT="${UNI_D2_ROOT}/outputs/block_qwen/sanity_${TAG}.json"

python tools/run_block_arm_sanity.py \
  --checkpoint "${CKPT}" \
  --device cuda \
  --batch-size 4 \
  --output "${OUT}" || RC=$?

echo "Wrote ${OUT} (exit ${RC:-0})"
exit "${RC:-0}"
