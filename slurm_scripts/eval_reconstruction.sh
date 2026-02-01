#!/bin/bash
#SBATCH --job-name=eval_reconstruction
#SBATCH --partition=accelerated-h200,accelerated-h100,accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/eval_reconstruction_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/eval_reconstruction_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
cd "${REPO_ROOT}"

module purge
module load compiler/gnu/13 || true
module load devel/cuda/12.4 || true

source venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src"

# Workspace paths
WORKSPACE_BASE="/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi"
CHECKPOINT_DIR="${WORKSPACE_BASE}/outputs/owt/stage2_decoder"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${WORKSPACE_BASE}/outputs/owt/reconstruction_eval/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORKSPACE_BASE}/logs"

# Use cached HuggingFace models
export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Reconstruction Quality Evaluation"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=========================================="

python scripts/eval_reconstruction.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --num_samples 10 \
    --save_samples \
    --seed 42 \
    --output_path "${OUTPUT_DIR}/reconstruction_results.json" \
    --device cuda

echo "=========================================="
echo "Reconstruction evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/reconstruction_results.json"
echo "=========================================="


