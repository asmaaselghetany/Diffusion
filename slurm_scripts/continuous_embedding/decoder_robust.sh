#!/bin/bash
#SBATCH --job-name=ced_dec_robust
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_dec_robust_%j.out
#SBATCH --error=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_dec_robust_%j.err

# ============================================================================
# Decoder-Robustness Stage-2 training (x0 / epsilon / v denoiser variants)
# - Frozen encoder + frozen denoiser
# - Train decoder CE on denoiser-predicted x0 latents
# - Inference-matched timestep policy
# ============================================================================

set -eo pipefail

REPO_ROOT="/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi"
OUTPUT_BASE="${REPO_ROOT}/outputs/continuous_embedding"
LOG_DIR="${OUTPUT_BASE}/logs"
DATA_CACHE_DIR="/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/datasets/pgm_owt"

DENOISER_KIND="${DENOISER_KIND:-x0}"  # x0 | epsilon | v
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-10000}"
MAX_STEPS="${MAX_STEPS:-100000}"
WANDB_MODE="${WANDB_MODE:-online}"

RESUME="${RESUME:-0}"                # 1/true to resume stage-2 run
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}" # explicit robust decoder run directory
RESUME_CKPT="${RESUME_CKPT:-}"       # explicit robust decoder checkpoint path

STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-}" # optional denoiser run directory
DENOISER_CKPT="${DENOISER_CKPT:-}"   # optional explicit denoiser checkpoint

case "${DENOISER_KIND}" in
    x0|epsilon|v) ;;
    *)
        echo "ERROR: DENOISER_KIND must be one of: x0, epsilon, v" >&2
        exit 2
        ;;
esac

EXPERIMENT_TYPE="decoder_robust_${DENOISER_KIND}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${RUN_NAME_OVERRIDE:-${EXPERIMENT_TYPE}_${TIMESTAMP}}"
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
    local prefix="$1"
    ls -1dt "${OUTPUT_BASE}/${prefix}_"* 2>/dev/null | head -n 1 || true
}

resolve_denoiser_ckpt() {
    if [[ -n "${DENOISER_CKPT}" && -f "${DENOISER_CKPT}" ]]; then
        echo "${DENOISER_CKPT}"
        return
    fi

    if [[ -z "${STAGE1_RUN_DIR}" ]]; then
        STAGE1_RUN_DIR="$(find_latest_run_dir "denoiser_${DENOISER_KIND}")"
    fi

    if [[ -n "${STAGE1_RUN_DIR}" ]]; then
        if [[ -f "${STAGE1_RUN_DIR}/checkpoints/last.ckpt" ]]; then
            echo "${STAGE1_RUN_DIR}/checkpoints/last.ckpt"
            return
        fi
        if [[ -f "${STAGE1_RUN_DIR}/checkpoints/best.ckpt" ]]; then
            echo "${STAGE1_RUN_DIR}/checkpoints/best.ckpt"
            return
        fi
    fi

    echo ""
}

cd "${REPO_ROOT}"

# Resolve resume checkpoint and reuse run directory when requested.
if [[ "$(to_bool "${RESUME}")" == "true" ]]; then
    if [[ -z "${RESUME_CKPT}" ]]; then
        if [[ -z "${RESUME_RUN_DIR}" ]]; then
            RESUME_RUN_DIR="$(find_latest_run_dir "${EXPERIMENT_TYPE}")"
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

SOURCE_DENOISER_CKPT="$(resolve_denoiser_ckpt)"
if [[ -z "${SOURCE_DENOISER_CKPT}" || ! -f "${SOURCE_DENOISER_CKPT}" ]]; then
    echo "ERROR: Could not resolve source denoiser checkpoint for ${DENOISER_KIND}." >&2
    echo "  STAGE1_RUN_DIR='${STAGE1_RUN_DIR}'" >&2
    echo "  DENOISER_CKPT='${DENOISER_CKPT}'" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${DATA_CACHE_DIR}" "${OUTPUT_DIR}/checkpoints"

# Ensure a stable last.ckpt entry point exists for future resumes.
if [[ "$(to_bool "${RESUME}")" == "true" && ! -f "${OUTPUT_DIR}/checkpoints/last.ckpt" ]]; then
    if [[ "${RESUME_CKPT_PATH}" != "${OUTPUT_DIR}/checkpoints/last.ckpt" ]]; then
        cp -f "${RESUME_CKPT_PATH}" "${OUTPUT_DIR}/checkpoints/last.ckpt"
    fi
fi

source ~/.bashrc || true
conda activate discrete_diffusion 2>/dev/null || source venv/bin/activate 2>/dev/null || true
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE}"

export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${DATA_CACHE_DIR}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${DATA_CACHE_DIR}/hf_hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

echo "=========================================="
echo "Decoder-Robustness Stage-2 Training - $(date)"
echo "Denoiser parameterization: ${DENOISER_KIND}"
echo "Source denoiser checkpoint: ${SOURCE_DENOISER_CKPT}"
echo "Output: ${OUTPUT_DIR}"
echo "Resume mode: $(to_bool "${RESUME}")"
echo "Resume checkpoint: ${RESUME_CKPT_PATH}"
echo "Checkpoint cadence: every ${CHECKPOINT_EVERY_N_STEPS} steps"
echo "WandB mode: ${WANDB_MODE}"
echo "=========================================="

srun python -m discrete_diffusion \
    algo=decoder_robust_${DENOISER_KIND} \
    model=continuous_embedding \
    data=openwebtext \
    data.cache_dir="${DATA_CACHE_DIR}" \
    noise=cosine \
    sampling=continuous_embedding \
    \
    training.finetune_path="${SOURCE_DENOISER_CKPT}" \
    \
    model.embed_dim=1024 \
    model.hidden_size=1024 \
    model.n_heads=16 \
    model.n_blocks=12 \
    model.decoder_n_blocks=4 \
    model.length=256 \
    model.gradient_checkpointing=false \
    model.encoder_name="Qwen/Qwen3-Embedding-0.6B" \
    model.encoder_dtype=bfloat16 \
    \
    loader.global_batch_size=64 \
    loader.batch_size=16 \
    loader.num_workers=4 \
    \
    trainer.devices=1 \
    trainer.max_steps=${MAX_STEPS} \
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
