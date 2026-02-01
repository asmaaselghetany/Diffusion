#!/bin/bash
#SBATCH --job-name=s2_teacher
#SBATCH --partition=accelerated
#SBATCH --reservation=llmtum
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/stage2_ablation_teacher_only_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/logs/stage2_ablation_teacher_only_%j.err

# ============================================================================
# ABLATION 1: TEACHER-ONLY LATENTS
# ============================================================================
# Hypothesis: Training decoder on clean teacher latents establishes upper bound
#             for decoder capacity. If this works well, the issue is predictor
#             latent quality. If it fails, the decoder itself needs improvement.
# Key change: latent_source=teacher (no mixing, pure teacher signal)
# ============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"
cd "${REPO_ROOT}"

module purge
module load compiler/gnu/13 || true
module load devel/cuda/12.4 || true

source venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src"
export WANDB_MODE=online

WORKSPACE_BASE="/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi"
STAGE1_CKPT="${WORKSPACE_BASE}/outputs/owt/stage1_encoder_0017_fixed_20260125_234231/checkpoints/last.ckpt"
OUTPUT_DIR="${WORKSPACE_BASE}/outputs/owt/stage2_ablation_teacher_only_$(date +%Y%m%d_%H%M%S)"
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
fi

CACHE_DIR="${REPO_ROOT}/.cache"
mkdir -p "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch_inductor"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"

echo "=========================================="
echo "ABLATION 1: TEACHER-ONLY LATENTS"
echo "Stage 1 Checkpoint: ${STAGE1_CKPT}"
echo "Latent Source: TEACHER (clean signal, upper bound test)"
echo "Batch: 256 global, 32 per-GPU | torch.compile=ON"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

DATA_CACHE="/home/hk-project-p0023960/hgf_nhz3359/New_Discrete_Diffusion-main/datasets/pgm_owt"

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
    algo.stage=2 \
    algo.latent_source=teacher \
    algo.sampling_eps=0.001 \
    training.torch_compile=true \
    training.finetune_path="'"$STAGE1_CKPT"'" \
    optim.lr=0.0003 \
    optim.weight_decay=0 \
    lr_scheduler.num_warmup_steps=1000 \
    loader.global_batch_size=256 \
    loader.num_workers=4 \
    trainer.num_nodes=${SLURM_NNODES} \
    trainer.devices=4 \
    trainer.accumulate_grad_batches=1 \
    trainer.max_steps=200000 \
    trainer.val_check_interval=10000 \
    trainer.log_every_n_steps=100 \
    trainer.precision=bf16-mixed \
    trainer.num_sanity_val_steps=2 \
    trainer.limit_val_batches=10 \
    eval.generate_samples=true \
    eval.save_validation_samples=true \
    eval.compute_perplexity_on_sanity=false \
    sampling.steps=64 \
    sampling.num_sample_batches=1 \
    sampling.num_sample_log=8 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=3 \
    callbacks.checkpoint_monitor.monitor=val/nll \
    callbacks.checkpoint_monitor.mode=min \
    callbacks.latent_eval.eval_frequency=10000 \
    callbacks.ema_teacher.update_frequency=10000 \
    strategy.find_unused_parameters=true \
    checkpointing.resume_from_ckpt=false \
    checkpointing.save_dir="'"$OUTPUT_DIR"'" \
    wandb.project=latent_jepa_stage2 \
    wandb.name=s2_ablation_teacher_only \
    wandb.group=stage2_decoder_ablation \
    wandb.save_dir="'"$OUTPUT_DIR"'/wandb" \
    hydra.run.dir="'"$OUTPUT_DIR"'"'

echo "Training completed at $(date)"
