#!/bin/bash
#SBATCH --job-name=ced_quick_test
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_quick_test_%j.out
#SBATCH --error=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_quick_test_%j.err

# ============================================================================
# Quick Test: Verify pipeline works end-to-end
# Small batch, few steps, minimal resources
# Run this FIRST before submitting full experiments
# ============================================================================

set -eo pipefail

# Repository and output paths
REPO_ROOT="/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi"
OUTPUT_BASE="${REPO_ROOT}/outputs/continuous_embedding"
EXPERIMENT_TYPE="quick_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${EXPERIMENT_TYPE}_${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
LOG_DIR="${OUTPUT_BASE}/logs"
DATA_CACHE_DIR="/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/datasets/pgm_owt"

cd "${REPO_ROOT}"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${DATA_CACHE_DIR}"

# Activate environment
source ~/.bashrc || true
conda activate discrete_diffusion 2>/dev/null || source venv/bin/activate 2>/dev/null || true
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# HuggingFace cache (force writable Lustre location)
export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${DATA_CACHE_DIR}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${DATA_CACHE_DIR}/hf_hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

echo "=========================================="
echo "Quick Pipeline Test - $(date)"
echo "Purpose: Verify all components work"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python -m discrete_diffusion \
    algo=continuous_embedding_diffusion \
    model=continuous_embedding \
    data=openwebtext \
    data.cache_dir="${DATA_CACHE_DIR}" \
    noise=cosine \
    sampling=continuous_embedding \
    \
    algo.stage=1 \
    algo.parameterization=x0 \
    algo.loss_type=simple \
    algo.lambda_ce=0.1 \
    \
    model.embed_dim=1024 \
    model.hidden_size=512 \
    model.n_heads=8 \
    model.n_blocks=4 \
    model.decoder_n_blocks=2 \
    model.length=256 \
    model.gradient_checkpointing=false \
    model.encoder_name="Qwen/Qwen3-Embedding-0.6B" \
    model.encoder_dtype=bfloat16 \
    \
    loader.global_batch_size=8 \
    loader.batch_size=8 \
    loader.num_workers=2 \
    \
    trainer.devices=1 \
    trainer.max_steps=100 \
    trainer.precision=bf16-mixed \
    trainer.val_check_interval=50 \
    trainer.log_every_n_steps=10 \
    trainer.num_sanity_val_steps=1 \
    trainer.limit_val_batches=2 \
    \
    optim.lr=1e-4 \
    \
    training.ema=0 \
    \
    eval.generate_samples=false \
    \
    checkpointing.save_dir="${OUTPUT_DIR}" \
    checkpointing.resume_from_ckpt=false \
    \
    wandb.project=continuous-embedding-diffusion \
    wandb.name="${RUN_NAME}" \
    wandb.save_dir="${OUTPUT_DIR}/wandb" \
    \
    hydra.run.dir="${OUTPUT_DIR}"

echo "=========================================="
echo "Quick test completed at $(date)"
echo "If you see this, the pipeline works!"
echo "=========================================="
