#!/bin/bash
#SBATCH --job-name=ced_train
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_train_%j.out
#SBATCH --error=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_train_%j.err

# ============================================================================
# Continuous Embedding Diffusion — unified training launcher
#
# Configurable via environment variables (sbatch --export=ALL,VAR=val ...):
#
#   OBJECTIVE       ddpm | flow_matching | flow_map | imf   (default: flow_map)
#   DATASET         openwebtext | lm1b                      (default: openwebtext)
#   NUM_GPUS        number of GPUs                          (default: 2)
#   MAX_STEPS       total optimizer steps                   (default: 100000)
#   GLOBAL_BATCH    global batch size                       (default: 128)
#   SEQ_LEN         sequence length                         (default: 256)
#   LR              learning rate                           (default: 1e-4)
#   LAMBDA_CE       CE auxiliary weight                     (default: 0.1)
#   S_ZERO_PROB     flow_map: fraction of s=0 samples       (default: 0.0)
#   CONDITIONING    none | span_masking                     (default: span_masking)
#   EMBED_PROVIDER  tied | lookup | legacy_contextual       (default: tied)
#   SAMPLING_METHOD sampler for eval generation             (default: flowmap_few_step)
#   CHECKPOINT_EVERY_N_STEPS                                (default: 5000)
#
# Sample generation (SampleSaver callback):
#   SAMPLE_EVERY_N_STEPS  generate samples every N steps    (default: 5000, 0=off)
#   NUM_GEN_SAMPLES       number of samples per generation  (default: 5)
#   NUM_GEN_STEPS         denoising steps for generation    (default: 8)
#
# Resume:
#   RESUME=1                   resume from latest run of same objective+dataset
#   RESUME_RUN_DIR=/path/...   resume from a specific run directory
#   RESUME_CKPT=/path/...ckpt  resume from an explicit checkpoint file
#
# Examples:
#   # Flow map on OWT (default)
#   sbatch train.sh
#
#   # Flow map on LM1B, 4 GPUs, longer run
#   sbatch --gres=gpu:4 --export=ALL,DATASET=lm1b,NUM_GPUS=4,MAX_STEPS=200000 train.sh
#
#   # Flow matching baseline for comparison
#   sbatch --export=ALL,OBJECTIVE=flow_matching,SAMPLING_METHOD=rectified_few_step train.sh
#
#   # iMF baseline
#   sbatch --export=ALL,OBJECTIVE=imf,SAMPLING_METHOD=rectified_few_step train.sh
#
#   # Quick smoke test (100 steps, 1 GPU)
#   sbatch --gres=gpu:1 --time=01:00:00 --export=ALL,NUM_GPUS=1,MAX_STEPS=100,GLOBAL_BATCH=8 train.sh
#
#   # Resume latest flow_map OWT run
#   sbatch --export=ALL,RESUME=1 train.sh
# ============================================================================

set -eo pipefail

# ---------------------------------------------------------------------------
# Parameters with defaults
# ---------------------------------------------------------------------------
OBJECTIVE="${OBJECTIVE:-flow_map}"
DATASET="${DATASET:-openwebtext}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_STEPS="${MAX_STEPS:-100000}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
SEQ_LEN="${SEQ_LEN:-256}"
LR="${LR:-1e-4}"
LAMBDA_CE="${LAMBDA_CE:-0.1}"
S_ZERO_PROB="${S_ZERO_PROB:-0.0}"
CONDITIONING="${CONDITIONING:-span_masking}"
EMBED_PROVIDER="${EMBED_PROVIDER:-tied}"
SAMPLING_METHOD="${SAMPLING_METHOD:-flowmap_few_step}"
TOP_P="${TOP_P:-0.9}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-5000}"
SAMPLE_EVERY_N_STEPS="${SAMPLE_EVERY_N_STEPS:-5000}"
NUM_GEN_SAMPLES="${NUM_GEN_SAMPLES:-5}"
NUM_GEN_STEPS="${NUM_GEN_STEPS:-8}"
WANDB_MODE="${WANDB_MODE:-online}"

RESUME="${RESUME:-0}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
RESUME_CKPT="${RESUME_CKPT:-}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi"
OUTPUT_BASE="${REPO_ROOT}/outputs/continuous_embedding"
LOG_DIR="${OUTPUT_BASE}/logs"
DATA_CACHE_DIR="/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/datasets/pgm_owt"

# Experiment naming: <objective>_<dataset>_<timestamp>
EXPERIMENT_TYPE="${OBJECTIVE}_${DATASET}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${EXPERIMENT_TYPE}_${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
RESUME_CKPT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"
RESUME_FROM_CKPT=true

# ---------------------------------------------------------------------------
# Derived: choose algo config and sampler based on objective
# ---------------------------------------------------------------------------
case "${OBJECTIVE}" in
    flow_map)
        ALGO_CONFIG="flow_map"
        ;;
    flow_matching|imf|meanflow|ddpm)
        ALGO_CONFIG="continuous_embedding_diffusion"
        ;;
    *)
        echo "ERROR: OBJECTIVE must be one of: flow_map, flow_matching, imf, ddpm" >&2
        exit 2
        ;;
esac

# Dataset-specific data config
case "${DATASET}" in
    openwebtext|lm1b) ;;
    *)
        echo "ERROR: DATASET must be one of: openwebtext, lm1b" >&2
        exit 2
        ;;
esac

# Per-GPU batch size (integer division, ceiling)
PER_GPU_BATCH=$(( (GLOBAL_BATCH + NUM_GPUS - 1) / NUM_GPUS ))

# SampleSaver: enabled when SAMPLE_EVERY_N_STEPS > 0
if [[ "${SAMPLE_EVERY_N_STEPS}" -gt 0 ]]; then
    SAMPLE_SAVER_ENABLED=true
else
    SAMPLE_SAVER_ENABLED=false
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
to_bool() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|y|Y) echo "true" ;;
        *) echo "false" ;;
    esac
}

find_latest_run_dir() {
    ls -1dt "${OUTPUT_BASE}/${EXPERIMENT_TYPE}_"* 2>/dev/null | head -n 1 || true
}

# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

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

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${DATA_CACHE_DIR}" "${OUTPUT_DIR}/checkpoints"

if [[ "$(to_bool "${RESUME}")" == "true" && ! -f "${OUTPUT_DIR}/checkpoints/last.ckpt" ]]; then
    if [[ "${RESUME_CKPT_PATH}" != "${OUTPUT_DIR}/checkpoints/last.ckpt" ]]; then
        cp -f "${RESUME_CKPT_PATH}" "${OUTPUT_DIR}/checkpoints/last.ckpt"
    fi
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
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

unset SLURM_MEM_PER_NODE || true
unset SLURM_MEM_PER_GPU || true

# ---------------------------------------------------------------------------
# Build objective-specific Hydra overrides
# ---------------------------------------------------------------------------
OBJECTIVE_OVERRIDES=()
case "${OBJECTIVE}" in
    flow_map)
        OBJECTIVE_OVERRIDES+=(
            "algo.objective=flow_map"
            "algo.flow_map.s_zero_prob=${S_ZERO_PROB}"
        )
        ;;
    flow_matching)
        OBJECTIVE_OVERRIDES+=(
            "algo.objective=flow_matching"
            "algo.interpolant=rectified"
        )
        ;;
    imf|meanflow)
        OBJECTIVE_OVERRIDES+=(
            "algo.objective=${OBJECTIVE}"
            "algo.interpolant=rectified"
        )
        ;;
    ddpm)
        OBJECTIVE_OVERRIDES+=(
            "algo.objective=ddpm"
            "algo.interpolant=vp"
        )
        ;;
esac

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Continuous Embedding Training — $(date)"
echo "  Objective:     ${OBJECTIVE}"
echo "  Dataset:       ${DATASET}"
echo "  Algo config:   ${ALGO_CONFIG}"
echo "  GPUs:          ${NUM_GPUS}"
echo "  Global batch:  ${GLOBAL_BATCH}  (per-GPU: ${PER_GPU_BATCH})"
echo "  Seq length:    ${SEQ_LEN}"
echo "  Max steps:     ${MAX_STEPS}"
echo "  LR:            ${LR}"
echo "  Lambda CE:     ${LAMBDA_CE}"
echo "  Conditioning:  ${CONDITIONING}"
echo "  Sampling:      ${SAMPLING_METHOD}"
echo "  Gen samples:   ${NUM_GEN_SAMPLES} every ${SAMPLE_EVERY_N_STEPS} steps (${NUM_GEN_STEPS}-step sampler)"
echo "  Output:        ${OUTPUT_DIR}"
echo "  Resume:        $(to_bool "${RESUME}")"
echo "  Resume ckpt:   ${RESUME_CKPT_PATH}"
echo "  WandB mode:    ${WANDB_MODE}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
srun python -m discrete_diffusion \
    algo="${ALGO_CONFIG}" \
    model=continuous_embedding \
    data="${DATASET}" \
    data.cache_dir="${DATA_CACHE_DIR}" \
    noise=cosine \
    sampling=continuous_embedding \
    \
    algo.stage=1 \
    algo.parameterization=x0 \
    algo.loss_type=simple \
    algo.lambda_ce="${LAMBDA_CE}" \
    algo.conditioning="${CONDITIONING}" \
    algo.embedding_provider="${EMBED_PROVIDER}" \
    algo.sampler.sampling_method="${SAMPLING_METHOD}" \
    algo.sampler.top_p="${TOP_P}" \
    "${OBJECTIVE_OVERRIDES[@]}" \
    \
    model.embed_dim=1024 \
    model.hidden_size=1024 \
    model.n_heads=16 \
    model.n_blocks=12 \
    model.decoder_n_blocks=2 \
    model.length="${SEQ_LEN}" \
    model.gradient_checkpointing=true \
    model.encoder_name="Qwen/Qwen3-Embedding-0.6B" \
    model.encoder_dtype=bfloat16 \
    model.parameterization=x0 \
    model.attn_backend=auto \
    \
    loader.global_batch_size="${GLOBAL_BATCH}" \
    loader.batch_size="${PER_GPU_BATCH}" \
    loader.num_workers=4 \
    \
    strategy.find_unused_parameters=true \
    \
    trainer.devices="${NUM_GPUS}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.precision=bf16-mixed \
    trainer.val_check_interval=2000 \
    trainer.log_every_n_steps=50 \
    trainer.num_sanity_val_steps=2 \
    \
    optim.lr="${LR}" \
    optim.weight_decay=0.01 \
    \
    training.ema=0.9999 \
    \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_every_n_steps.every_n_train_steps="${CHECKPOINT_EVERY_N_STEPS}" \
    callbacks.checkpoint_monitor.save_last=true \
    callbacks.sample_saver.enabled="${SAMPLE_SAVER_ENABLED}" \
    callbacks.sample_saver.every_n_steps="${SAMPLE_EVERY_N_STEPS}" \
    callbacks.sample_saver.num_samples="${NUM_GEN_SAMPLES}" \
    callbacks.sample_saver.num_steps="${NUM_GEN_STEPS}" \
    callbacks.sample_saver.save_dir="${OUTPUT_DIR}/samples" \
    callbacks.sample_saver.log_to_wandb=true \
    \
    checkpointing.save_dir="${OUTPUT_DIR}" \
    checkpointing.resume_from_ckpt="${RESUME_FROM_CKPT}" \
    checkpointing.resume_ckpt_path="${RESUME_CKPT_PATH}" \
    \
    eval.generate_samples=false \
    \
    wandb.project=continuous-embedding-diffusion \
    wandb.name="${RUN_NAME}" \
    wandb.tags="[${OBJECTIVE},${DATASET}]" \
    wandb.save_dir="${OUTPUT_DIR}/wandb" \
    \
    hydra.run.dir="${OUTPUT_DIR}"

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
