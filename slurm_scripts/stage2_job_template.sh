#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --reservation=llmtum
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

set -euo pipefail

# Parse positional arguments
STAGE1_CKPT_PATH="$1"
OUTPUT_DIR="$2"
RUN_NAME="$3"
MODEL_LATENT_DIM="$4"
MODEL_HIDDEN_SIZE="$5"
MODEL_N_HEADS="$6"
MODEL_N_BLOCKS="$7"
MODEL_PREDICTOR_DEPTH="$8"
MODEL_PREDICTOR_HIDDEN="$9"
MODEL_PREDICTOR_HEADS="${10}"
MODEL_READOUT_TYPE="${11}"
MODEL_READOUT_DEPTH="${12}"
MODEL_READOUT_HIDDEN="${13}"
MODEL_TIME_EMBED_DIM="${14}"
MODEL_DROPOUT="${15}"
MODEL_PREDICTOR_USE_PROJ="${16:-false}"

REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
WORKSPACE_BASE="/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi"

cd "${REPO_ROOT}"

module purge
module load compiler/gnu/13 || true
module load devel/cuda/12.4 || true

source venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src"
export WANDB_MODE=online

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORKSPACE_BASE}/logs"

export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1

export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_FAMILY=AF_INET

# Auto-detect network interface
RDZV_IFNAME=""
for iface in ib0 ipogif0 eth0; do
  if ip link show dev "${iface}" >/dev/null 2>&1; then
    if ip -4 addr show dev "${iface}" 2>/dev/null | grep -q "inet "; then
      RDZV_IFNAME="${iface}"; break
    fi
  fi
done
if [[ -z "${RDZV_IFNAME}" ]]; then
  for iface in $(ip -o link show | awk -F': ' '{print $2}' | grep -E '^(enp|eno|ens)'); do
    if ip -4 addr show dev "${iface}" 2>/dev/null | grep -q "inet "; then
      RDZV_IFNAME="${iface}"; break
    fi
  done
fi
[[ -n "${RDZV_IFNAME}" ]] && export GLOO_SOCKET_IFNAME="${RDZV_IFNAME}" NCCL_SOCKET_IFNAME="${RDZV_IFNAME}"

CACHE_DIR="${REPO_ROOT}/.cache"
mkdir -p "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch_inductor"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"

echo "=========================================="
echo "Stage-2 Decoder Training: ${RUN_NAME}"
echo "Checkpoint: ${STAGE1_CKPT_PATH}"
echo "Nodes: ${SLURM_NNODES}, GPUs per node: 4"
echo "Config: latent=${MODEL_LATENT_DIM} hidden=${MODEL_HIDDEN_SIZE} heads=${MODEL_N_HEADS} blocks=${MODEL_N_BLOCKS}"
echo "=========================================="

DATA_CACHE_SHARED="/home/hk-project-p0023960/hgf_nhz3359/New_Discrete_Diffusion-main/datasets/pgm_owt"
DATA_CACHE="${DATA_CACHE_SHARED}"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  DATA_CACHE="${SLURM_TMPDIR}/data_cache"
  srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --export=ALL bash -c '
set -euo pipefail
mkdir -p "'"${DATA_CACHE}"'"
for d in openwebtext-train_train_bs1024_wrapped.dat openwebtext-valid_validation_bs1024_wrapped.dat; do
  src="'"${DATA_CACHE_SHARED}"'/${d}"
  [[ -e "${src}" ]] || continue
  cp -a "${src}" "'"${DATA_CACHE}"'/" || true
done
'
fi
mkdir -p "${DATA_CACHE}"

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR_TMP=""
if [[ -n "${RDZV_IFNAME:-}" ]]; then
  MASTER_ADDR_TMP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" ip -4 -o addr show dev "${RDZV_IFNAME}" 2>/dev/null | head -n 1 | awk '{print $4}' | cut -d/ -f1)
fi
if [[ -z "${MASTER_ADDR_TMP}" ]]; then
  MASTER_ADDR_TMP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname -i 2>/dev/null | awk '{print $1}')
fi
export MASTER_ADDR="${MASTER_ADDR_TMP}"
export MASTER_PORT=29500

[[ -z "${MASTER_ADDR}" ]] && { echo "ERROR: Could not determine MASTER_ADDR"; exit 1; }
echo "Master: ${MASTER_NODE} -> (${MASTER_ADDR}:${MASTER_PORT})"

export DATA_CACHE MASTER_ADDR MASTER_PORT

srun --kill-on-bad-exit=1 --export=ALL bash -c 'torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=4 \
    --node_rank=${SLURM_NODEID} \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m discrete_diffusion \
    data=openwebtext-split \
    data.cache_dir="'"$DATA_CACHE"'" \
    model=latent_jepa_180 \
    model.length=1024 \
    model.gradient_checkpointing=false \
    model.latent_dim='"$MODEL_LATENT_DIM"' \
    model.hidden_size='"$MODEL_HIDDEN_SIZE"' \
    model.n_heads='"$MODEL_N_HEADS"' \
    model.n_blocks='"$MODEL_N_BLOCKS"' \
    model.predictor_depth='"$MODEL_PREDICTOR_DEPTH"' \
    model.predictor_hidden_size='"$MODEL_PREDICTOR_HIDDEN"' \
    model.predictor_n_heads='"$MODEL_PREDICTOR_HEADS"' \
    model.predictor_use_projections='"$MODEL_PREDICTOR_USE_PROJ"' \
    model.time_embed_dim='"$MODEL_TIME_EMBED_DIM"' \
    model.dropout='"$MODEL_DROPOUT"' \
    algo=jepa \
    algo.stage=2 \
    training.torch_compile=false \
    loader.batch_size=16 \
    loader.global_batch_size=128 \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes=${SLURM_NNODES} \
    trainer.devices=4 \
    trainer.accumulate_grad_batches=1 \
    trainer.val_check_interval=10000 \
    trainer.log_every_n_steps=100 \
    trainer.precision=bf16-mixed \
    trainer.num_sanity_val_steps=0 \
    trainer.limit_val_batches=10 \
    eval.generate_samples=true \
    eval.save_validation_samples=true \
    eval.compute_perplexity_on_sanity=false \
    sampling.steps=64 \
    sampling.num_sample_batches=1 \
    sampling.num_sample_log=8 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.monitor=val/nll \
    callbacks.checkpoint_monitor.save_top_k=3 \
    strategy.find_unused_parameters=true \
    checkpointing.resume_from_ckpt=true \
    training.finetune_path="'"$STAGE1_CKPT_PATH"'" \
    checkpointing.save_dir="'"$OUTPUT_DIR"'" \
    wandb.project=latent_jepa_stage2 \
    wandb.name=s2_'"$RUN_NAME"' \
    wandb.save_dir="'"$OUTPUT_DIR"'/wandb" \
    hydra.run.dir="'"$OUTPUT_DIR"'"'

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
