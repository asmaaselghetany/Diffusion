#!/bin/bash
# Stage 2: Decoder-only fine-tuning for Continuous Embedding Diffusion
# Requires a Stage 1 checkpoint

CHECKPOINT=${CHECKPOINT:-"path/to/stage1/checkpoint.ckpt"}
DEVICES=${DEVICES:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-50000}

python -m discrete_diffusion \
    algo=continuous_embedding_diffusion \
    model=continuous_embedding \
    data=openwebtext \
    noise=cosine \
    sampling=continuous_embedding \
    \
    algo.stage=2 \
    \
    model.embed_dim=1024 \
    model.hidden_size=1024 \
    model.n_heads=16 \
    model.n_blocks=12 \
    model.decoder_n_blocks=2 \
    model.length=256 \
    model.gradient_checkpointing=true \
    model.encoder_name="Qwen/Qwen3-Embedding-0.6B" \
    model.encoder_dtype=bfloat16 \
    \
    loader.global_batch_size=${BATCH_SIZE} \
    trainer.devices=${DEVICES} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.precision=bf16 \
    trainer.val_check_interval=2000 \
    \
    optim.lr=5e-5 \
    optim.weight_decay=0.01 \
    \
    training.ema=0.9999 \
    training.finetune_path="${CHECKPOINT}" \
    \
    wandb.project=continuous-embedding-diffusion \
    wandb.name="ced-owt-stage2"
