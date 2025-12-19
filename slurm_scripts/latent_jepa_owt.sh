#!/bin/bash
#SBATCH --job-name=jepa_owt
#SBATCH --partition=accelerated-h200,accelerated-h100,accelerated
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
##SBATCH --reservation=llmtum
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=logs/jepa_owt_%j.out
#SBATCH --error=logs/jepa_owt_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
cd "${REPO_ROOT}"

# module purge
# module load Stages/2025
# module load GCCcore/.13.3.0
# module load Python/3.12.3
# module load CUDA/12

module purge
module load compiler/gnu/13 || true
module load devel/cuda/12.4 || true


source venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src"
export WANDB_MODE=offline

# Use cached HuggingFace models (no internet on compute nodes)
# Use existing HuggingFace cache from home directory
export HF_HOME="/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# DEPRECATED: export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1

# NCCL settings for InfiniBand
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_FAMILY=AF_INET

# Network Interfaces - auto-detect a valid interface
# Try common HPC interface patterns, fall back to first available with IP
RDZV_IFNAME=""
for iface in ib0 ipogif0 eth0; do
  if ip link show dev "${iface}" >/dev/null 2>&1; then
    if ip -4 addr show dev "${iface}" 2>/dev/null | grep -q "inet "; then
      RDZV_IFNAME="${iface}"
      break
    fi
  fi
done
# Try wildcard patterns if no standard interface found
if [[ -z "${RDZV_IFNAME}" ]]; then
  for iface in $(ip -o link show | awk -F': ' '{print $2}' | grep -E '^(enp|eno|ens)'); do
    if ip -4 addr show dev "${iface}" 2>/dev/null | grep -q "inet "; then
      RDZV_IFNAME="${iface}"
      break
    fi
  done
fi

# If interface found, set NCCL/GLOO to use it
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
echo "Latent JEPA OWT Training - $(date)"
echo "Nodes: ${SLURM_NNODES}, GPUs per node: 4"
echo "=========================================="

DATA_CACHE_SHARED="${REPO_ROOT}/data_cache"
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

# Get master node's IP - use hostname resolution (more portable)
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Try to get IP from detected interface, fall back to hostname -i
MASTER_ADDR_TMP=""
if [[ -n "${RDZV_IFNAME}" ]]; then
  MASTER_ADDR_TMP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" ip -4 -o addr show dev "${RDZV_IFNAME}" 2>/dev/null | head -n 1 | awk '{print $4}' | cut -d/ -f1)
fi
if [[ -z "${MASTER_ADDR_TMP}" ]]; then
  # Fallback: resolve hostname to IP
  MASTER_ADDR_TMP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname -i 2>/dev/null | awk '{print $1}')
fi
export MASTER_ADDR="${MASTER_ADDR_TMP}"
export MASTER_PORT=29500

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "ERROR: Could not determine MASTER_ADDR"
  exit 1
fi
echo "Master: ${MASTER_NODE} -> (${MASTER_ADDR}:${MASTER_PORT})"

# Export for srun subprocesses
export DATA_CACHE MASTER_ADDR MASTER_PORT

# Multi-node training with torchrun per node
srun --kill-on-bad-exit=1 --export=ALL bash -c 'torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=4 \
    --node_rank=${SLURM_NODEID} \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m discrete_diffusion \
    data=openwebtext-split \
    data.cache_dir="${DATA_CACHE}" \
    model=latent_jepa_350 \
    model.length=1024 \
    algo=jepa \
    algo.stage=1 \
    training.torch_compile=false \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes=${SLURM_NNODES} \
    trainer.devices=4 \
    trainer.val_check_interval=1000 \
    trainer.log_every_n_steps=100 \
    trainer.precision=bf16-mixed \
    trainer.num_sanity_val_steps=0 \
    trainer.limit_val_batches=0 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=-1 \
    strategy.find_unused_parameters=true \
    checkpointing.resume_from_ckpt=false \
    wandb.project=latent_jepa \
    wandb.name=jepa_owt \
    hydra.run.dir=./outputs/owt/latent_jepa'

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="

