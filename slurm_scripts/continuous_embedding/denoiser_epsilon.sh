#!/bin/bash
#SBATCH --job-name=ced_epsilon
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_epsilon_%j.out
#SBATCH --error=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_epsilon_%j.err

# ============================================================================
# Epsilon Parameterization: Denoiser predicts noise (DDPM-style)
# Standard image diffusion approach, may need different loss weighting
# ============================================================================

set -eo pipefail

# Repository and output paths
REPO_ROOT="/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi"
OUTPUT_BASE="${REPO_ROOT}/outputs/continuous_embedding"
EXPERIMENT_TYPE="denoiser_epsilon"
LOG_DIR="${OUTPUT_BASE}/logs"
DATA_CACHE_DIR="/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/datasets/pgm_owt"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-5000}"
RESUME="${RESUME:-0}"                     # 1/true to resume
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"      # optional explicit run dir
RESUME_CKPT="${RESUME_CKPT:-}"            # optional explicit checkpoint path

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${EXPERIMENT_TYPE}_${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
RESUME_CKPT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"
RESUME_FROM_CKPT=true

to_bool() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|y|Y) echo "true" ;;
        *) echo "false" ;;
    esac
}

find_latest_run_dir() {
    ls -1dt "${OUTPUT_BASE}/${EXPERIMENT_TYPE}_"* 2>/dev/null | head -n 1 || true
}

cd "${REPO_ROOT}"

# Resolve resume checkpoint and reuse run directory when requested.
if [[ "$(to_bool "${RESUME}")" == "true" ]]; then
    if [[ -z "${RESUME_CKPT}" ]]; then
        if [[ -z "${RESUME_RUN_DIR}" ]]; then
            RESUME_RUN_DIR="$(find_latest_run_dir)"
        fi
        if [[ -n "${RESUME_RUN_DIR}" ]]; then
            if [[ -f "${RESUME_RUN_DIR}/checkpoints/last.ckpt" ]]; then
                RESUME_CKPT="${RESUME_RUN_DIR}/checkpoints/last.ckpt"
            elif [[ -f "${RESUME_RUN_DIR}/checkpoints/best.ckpt" ]]; then
                RESUME_CKPT="${RESUME_RUN_DIR}/checkpoints/best.ckpt"
            fi
        fi
    fi

    if [[ -z "${RESUME_CKPT}" || ! -f "${RESUME_CKPT}" ]]; then
        echo "ERROR: resume requested but no checkpoint found." >&2
        echo "  RESUME_RUN_DIR='${RESUME_RUN_DIR}'" >&2
        echo "  RESUME_CKPT='${RESUME_CKPT}'" >&2
        exit 2
    fi

    if [[ -z "${RESUME_RUN_DIR}" ]]; then
        RESUME_RUN_DIR="$(cd "$(dirname "${RESUME_CKPT}")/.." && pwd)"
    fi

    OUTPUT_DIR="${RESUME_RUN_DIR}"
    RUN_NAME="$(basename "${OUTPUT_DIR}")"
    RESUME_CKPT_PATH="${RESUME_CKPT}"
fi

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${DATA_CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"

# Ensure a stable last.ckpt entry point exists for future resumes.
# Use copy (not symlink) so writes to last.ckpt never mutate best.ckpt.
if [[ "$(to_bool "${RESUME}")" == "true" && ! -f "${OUTPUT_DIR}/checkpoints/last.ckpt" ]]; then
    if [[ "${RESUME_CKPT_PATH}" != "${OUTPUT_DIR}/checkpoints/last.ckpt" ]]; then
        cp -f "${RESUME_CKPT_PATH}" "${OUTPUT_DIR}/checkpoints/last.ckpt"
    fi
fi

# Activate environment
source ~/.bashrc || true
conda activate discrete_diffusion 2>/dev/null || source venv/bin/activate 2>/dev/null || true
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

# HuggingFace cache (force writable Lustre location)
export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${DATA_CACHE_DIR}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${DATA_CACHE_DIR}/hf_hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

NUM_GPUS=2

# Avoid inherited parent-allocation memory env collisions in nested sbatch->srun flows.
unset SLURM_MEM_PER_NODE || true
unset SLURM_MEM_PER_GPU || true

echo "=========================================="
echo "Epsilon Parameterization Training - $(date)"
echo "Denoiser predicts: noise epsilon"
echo "GPUs: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "Resume mode: $(to_bool "${RESUME}")"
echo "Resume checkpoint: ${RESUME_CKPT_PATH}"
echo "WandB mode: ${WANDB_MODE}"
echo "=========================================="

# Use srun for proper SLURM GPU allocation
srun python -m discrete_diffusion \
    algo=continuous_embedding_diffusion \
    model=continuous_embedding \
    data=openwebtext \
    data.cache_dir="${DATA_CACHE_DIR}" \
    noise=cosine \
    sampling=continuous_embedding \
    \
    algo.stage=1 \
    algo.parameterization=epsilon \
    algo.loss_type=simple \
    \
    model.embed_dim=1024 \
    model.hidden_size=1024 \
    model.n_heads=16 \
    model.n_blocks=12 \
    model.decoder_n_blocks=2 \
    model.length=256 \
    model.gradient_checkpointing=false \
    model.encoder_name="Qwen/Qwen3-Embedding-0.6B" \
    model.encoder_dtype=bfloat16 \
    model.parameterization=epsilon \
    \
    loader.global_batch_size=128 \
    loader.batch_size=32 \
    loader.num_workers=4 \
    \
    trainer.devices=${NUM_GPUS} \
    trainer.max_steps=100000 \
    trainer.precision=bf16-mixed \
    trainer.val_check_interval=2000 \
    trainer.log_every_n_steps=50 \
    trainer.num_sanity_val_steps=2 \
    \
    optim.lr=1e-4 \
    optim.weight_decay=0.01 \
    \
    training.ema=0.9999 \
    \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS} \
    callbacks.checkpoint_monitor.save_last=true \
    \
    checkpointing.save_dir="${OUTPUT_DIR}" \
    checkpointing.resume_from_ckpt=${RESUME_FROM_CKPT} \
    checkpointing.resume_ckpt_path="${RESUME_CKPT_PATH}" \
    \
    wandb.project=continuous-embedding-diffusion \
    wandb.name="${RUN_NAME}" \
    wandb.save_dir="${OUTPUT_DIR}/wandb" \
    \
    hydra.run.dir="${OUTPUT_DIR}"

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
