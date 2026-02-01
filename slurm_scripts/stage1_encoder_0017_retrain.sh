#!/bin/bash
#SBATCH --job-name=s1_enc0017_opt
#SBATCH --partition=accelerated
#SBATCH --reservation=llmtum
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/stage1_encoder_0017_opt_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/stage1_encoder_0017_opt_%j.err

# Stage 1 JEPA Training - OPTIMIZED for efficiency
# Configuration: latent_dim=256, hidden_size=768, n_heads=4, n_blocks=8
# OPTIMIZATIONS:
#   - Reduced from 14 nodes to 4 nodes (16 GPUs) - better compute/communication ratio
#   - Increased per-GPU batch from 6 to 32 - better GPU utilization
#   - Disabled gradient checkpointing - not needed for 235M model
#   - Enabled torch.compile - free 10-30% speedup
#   - find_unused_parameters=true required (teacher encoder not in loss graph)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
cd "${REPO_ROOT}"

module purge
module load compiler/gnu/13 || true
module load devel/cuda/12.4 || true

source venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src"
export WANDB_MODE=online

# Resume from existing checkpoint directory
WORKSPACE_BASE="/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi"
export OUTPUT_DIR="${WORKSPACE_BASE}/outputs/owt/stage1_encoder_0017_fixed_20260125_234231"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORKSPACE_BASE}/logs"

# Use cached HuggingFace models (no internet on compute nodes)
export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1

# NCCL settings for InfiniBand
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_FAMILY=AF_INET

# Network Interfaces - auto-detect a valid interface
RDZV_IFNAME=""
for iface in ib0 ipogif0 eth0; do
  if ip link show dev "${iface}" >/dev/null 2>&1; then
    if ip -4 addr show dev "${iface}" 2>/dev/null | grep -q "inet "; then
      RDZV_IFNAME="${iface}"
      break
    fi
  fi
done
if [[ -z "${RDZV_IFNAME}" ]]; then
  for iface in $(ip -o link show | awk -F': ' '{print $2}' | grep -E '^(enp|eno|ens)'); do
    if ip -4 addr show dev "${iface}" 2>/dev/null | grep -q "inet "; then
      RDZV_IFNAME="${iface}"
      break
    fi
  done
fi

if [[ -n "${RDZV_IFNAME}" ]]; then
  export GLOO_SOCKET_IFNAME="${RDZV_IFNAME}"
  export NCCL_SOCKET_IFNAME="${RDZV_IFNAME}"
  echo "Using network interface: ${RDZV_IFNAME}"
fi

# Set all cache dirs to project dir (to avoid home quota issues)
CACHE_DIR="${REPO_ROOT}/.cache"
mkdir -p "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch_inductor"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"

echo "=========================================="
echo "Stage 1 JEPA Training (OPTIMIZED) - $(date)"
echo "Config: latent_dim=256, hidden_size=768, n_heads=4, n_blocks=8"
echo "Nodes: ${SLURM_NNODES}, GPUs per node: 4, Total: $((SLURM_NNODES * 4)) GPUs"
echo "Batch: 512 global, 32 per-GPU | torch.compile=ON | grad_ckpt=OFF"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Data cache setup
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

# Get master node's IP
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR_TMP=""
if [[ -n "${RDZV_IFNAME}" ]]; then
  MASTER_ADDR_TMP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" ip -4 -o addr show dev "${RDZV_IFNAME}" 2>/dev/null | head -n 1 | awk '{print $4}' | cut -d/ -f1)
fi
if [[ -z "${MASTER_ADDR_TMP}" ]]; then
  MASTER_ADDR_TMP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname -i 2>/dev/null | awk '{print $1}')
fi
export MASTER_ADDR="${MASTER_ADDR_TMP}"
export MASTER_PORT=29500

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "ERROR: Could not determine MASTER_ADDR"
  exit 1
fi
echo "Master: ${MASTER_NODE} -> (${MASTER_ADDR}:${MASTER_PORT})"

export DATA_CACHE MASTER_ADDR MASTER_PORT

# ====================================================
# Stage 1 Training with FIXED normalization
# Config matches encoder_0017 exactly
# ====================================================
srun --kill-on-bad-exit=1 --export=ALL bash -c 'torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=4 \
    --node_rank=${SLURM_NODEID} \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m discrete_diffusion \
    data=openwebtext-split \
    data.cache_dir="'"$DATA_CACHE"'" \
    model=latent_jepa_180M \
    model.length=1024 \
    model.latent_dim=256 \
    model.hidden_size=768 \
    model.n_heads=4 \
    model.n_blocks=8 \
    model.dropout=0.1 \
    model.time_embed_dim=256 \
    model.predictor_type=transformer \
    model.predictor_depth=6 \
    model.predictor_hidden_size=384 \
    model.predictor_n_heads=6 \
    model.predictor_use_projections=true \
    model.readout_type=tiny_transformer \
    model.readout_hidden_size=512 \
    model.readout_depth=2 \
    model.gradient_checkpointing=false \
    algo=jepa \
    algo.stage=1 \
    algo.loss_norm=l2 \
    algo.mask_only=true \
    algo.normalize_targets=true \
    algo.redundancy=vicreg \
    algo.lambda_var=0.1 \
    algo.lambda_cov=0.04 \
    algo.vicreg_eps=0.001 \
    algo.sampling_eps=0.001 \
    training.torch_compile=true \
    optim.lr=0.0003 \
    optim.weight_decay=0 \
    lr_scheduler.num_warmup_steps=2500 \
    loader.global_batch_size=512 \
    loader.num_workers=8 \
    trainer.num_nodes=${SLURM_NNODES} \
    trainer.devices=4 \
    trainer.accumulate_grad_batches=1 \
    trainer.max_steps=200000 \
    trainer.val_check_interval=10000 \
    trainer.log_every_n_steps=100 \
    trainer.precision=bf16 \
    trainer.num_sanity_val_steps=2 \
    trainer.limit_val_batches=1.0 \
    eval.generate_samples=true \
    eval.save_validation_samples=false \
    eval.compute_perplexity_on_sanity=false \
    sampling.steps=5000 \
    sampling.num_sample_batches=1 \
    sampling.num_sample_log=2 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=3 \
    callbacks.checkpoint_monitor.monitor=val/nll \
    callbacks.checkpoint_monitor.mode=min \
    strategy.find_unused_parameters=true \
    checkpointing.resume_from_ckpt=true \
    checkpointing.save_dir="'"$OUTPUT_DIR"'" \
    wandb.project=latent_jepa \
    wandb.name=stage1_encoder_0017_fixed_norm \
    wandb.group=stage1_fixed_normalization \
    wandb.save_dir="'"$OUTPUT_DIR"'/wandb" \
    hydra.run.dir="'"$OUTPUT_DIR"'"'

echo "=========================================="
echo "Training completed at $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

