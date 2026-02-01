#!/bin/bash
#SBATCH --job-name=eval_mdlm_gen_ppl
#SBATCH --partition=accelerated-h200,accelerated-h100,accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/eval_mdlm_gen_ppl_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/eval_mdlm_gen_ppl_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
cd "${REPO_ROOT}"

module purge
module load compiler/gnu/13 || true
module load devel/cuda/12.4 || true

source venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src:/home/hk-project-p0023960/hgf_nhz3359/New_Discrete_Diffusion-main/src"

# Workspace paths
WORKSPACE_BASE="/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/hk-project-p0023960/hgf_nhz3359/New_Discrete_Diffusion-main/outputs/owt/mdlm/dummy_checkpoints/checkpoints/best.ckpt}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${WORKSPACE_BASE}/outputs/owt/mdlm_gen_ppl_eval/${TIMESTAMP}_unconditional"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORKSPACE_BASE}/logs"

# Use cached HuggingFace models
export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Starting MDLM Generative PPL Evaluation"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Mode: Unconditional"
echo "=========================================="

python scripts/eval_mdlm_gen_ppl.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_samples 10 \
    --num_steps 256 \
    --eval_model gpt2-xl \
    --batch_size 4 \
    --save_samples \
    --prompt_mode none \
    --output_path "${OUTPUT_DIR}/gen_ppl_results.json" \
    --device cuda

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/gen_ppl_results.json"
echo "=========================================="
