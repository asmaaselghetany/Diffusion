#!/bin/bash
#SBATCH --job-name=jepa_owt
#SBATCH --partition=booster
#SBATCH --account=bacprot
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=jepa_owt_%j.out
#SBATCH --error=jepa_owt_%j.err

set -euo pipefail

REPO_ROOT="/p/project1/bacprot/kalyan/UNI-D2"
cd "${REPO_ROOT}"

module purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3
module load CUDA/12

source venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src"
export WANDB_MODE=offline

# Use cached HuggingFace models (no internet on compute nodes)
export HF_HOME="${REPO_ROOT}/data_cache/hf_cache"
export HF_DATASETS_CACHE="${REPO_ROOT}/data_cache/hf_datasets"
export TRANSFORMERS_CACHE="${REPO_ROOT}/data_cache/hf_transformers"
export HF_HUB_OFFLINE=1

# NCCL settings for InfiniBand
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_FAMILY=AF_INET

# Network Interfaces (prefer routable IB for rendezvous + collectives)
RDZV_IFNAME=ib0
if ! ip link show dev "${RDZV_IFNAME}" >/dev/null 2>&1; then
  RDZV_IFNAME=enp225s0f0
fi
export GLOO_SOCKET_IFNAME="${RDZV_IFNAME}"
export NCCL_SOCKET_IFNAME="${RDZV_IFNAME}"

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

# Get master node's IP (use same interface as rendezvous)
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" ip -4 -o addr show dev "${RDZV_IFNAME}" | head -n 1 | awk '{print $4}' | cut -d/ -f1)
export MASTER_PORT=29500
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
    model=latent_jepa \
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

