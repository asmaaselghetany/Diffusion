#!/bin/bash
#SBATCH --job-name=eval_gen_ppl_prefix
#SBATCH --partition=accelerated-h200,accelerated-h100,accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/eval_gen_ppl_prefix_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/eval_gen_ppl_prefix_%j.err

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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORKSPACE_BASE}/outputs/owt/stage2_decoder}"
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN:-*/checkpoints/best.ckpt}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${WORKSPACE_BASE}/outputs/owt/gen_ppl_eval/${TIMESTAMP}_prefix"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORKSPACE_BASE}/logs"

# Use cached HuggingFace models
export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Stage 2 Gen-PPL Evaluation: PREFIX (Left-to-Right) Mode"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Checkpoint pattern: ${CHECKPOINT_PATTERN}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Sampling config: configs/sampling/latent_jepa_prefix.yaml"
echo "=========================================="

# Prefix (left-to-right completion) mode
# The model is given a prompt and must complete it
# Sampling parameters loaded from config file
python scripts/eval_stage2_gen_ppl.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --checkpoint_pattern "${CHECKPOINT_PATTERN}" \
    --sampling_config configs/sampling/latent_jepa_prefix.yaml \
    --num_samples 10 \
    --eval_model gpt2-xl \
    --batch_size 4 \
    --save_samples \
    --output_path "${OUTPUT_DIR}/gen_ppl_results_prefix.json" \
    --device cuda

echo "=========================================="
echo "Prefix mode evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/gen_ppl_results_prefix.json"
echo "=========================================="
