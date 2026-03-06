#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

python -u -m discrete_diffusion \
    data=tiny_shakespeare \
    model=small \
    model.length=128 \
    algo=mdlm \
    loader.global_batch_size=64 \
    loader.eval_global_batch_size=64 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.max_steps=2000 \
    trainer.val_check_interval=200 \
    trainer.log_every_n_steps=50 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=500 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    checkpointing.resume_from_ckpt=false \
    wandb.project="toy_runs" \
    wandb.name="mdlm_tiny_shakespeare" \
    hydra.run.dir=./outputs/tiny_shakespeare/mdlm
