#!/usr/bin/env bash
# Local smoke run for unified block diffusion on Tiny Shakespeare.
# Usage:
#   MODE=masked ./examples/block_diffusion/tiny_shakespeare.sh
#   MODE=uniform ./examples/block_diffusion/tiny_shakespeare.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE=${MODE:-masked}
BLOCK_SIZE=${BLOCK_SIZE:-16}
MAX_STEPS=${MAX_STEPS:-200}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
SEQ_LEN=${SEQ_LEN:-128}
DATA_CACHE_DIR=${DATA_CACHE_DIR:-./data_cache}

if [[ "${MODE}" == "uniform" ]]; then
  ALGO=block_diffusion_uniform
else
  ALGO=block_diffusion_masked
fi

export BLOCK_SIZE
export PYTHONPATH="src:${PYTHONPATH:-}"

python -u -m discrete_diffusion \
    data=tiny_shakespeare \
    data.cache_dir="${DATA_CACHE_DIR}" \
    model=block_dit \
    model.length="${SEQ_LEN}" \
    algo="${ALGO}" \
    sampling=block_diffusion \
    block_size="${BLOCK_SIZE}" \
    loader.global_batch_size="${GLOBAL_BATCH}" \
    loader.batch_size="${GLOBAL_BATCH}" \
    loader.num_workers=2 \
    trainer.devices=1 \
    trainer.accelerator=cpu \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.val_check_interval=100 \
    trainer.log_every_n_steps=50 \
    trainer.num_sanity_val_steps=1 \
    training.resample=True \
    algo.ignore_bos=True \
    model.adaln=False \
    model.tie_word_embeddings=True \
    model.attn_backend=sdpa \
    eval.generate_samples=False \
    checkpointing.resume_from_ckpt=false \
    wandb=null \
    hydra.run.dir="./outputs/tiny_shakespeare/${ALGO}_block${BLOCK_SIZE}"
