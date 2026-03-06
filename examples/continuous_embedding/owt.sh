#!/bin/bash
# Continuous Embedding Diffusion training on OpenWebText
# Uses frozen Qwen3-Embedding-0.6B as encoder

# Default settings - adjust based on available resources
DEVICES=${DEVICES:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-100000}

python -m discrete_diffusion \
    algo=continuous_embedding_diffusion \
    model=continuous_embedding \
    data=openwebtext \
    noise=cosine \
    sampling=continuous_embedding \
    \
    algo.stage=1 \
    algo.parameterization=x0 \
    algo.loss_type=simple \
    algo.lambda_ce=0.1 \
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
    optim.lr=1e-4 \
    optim.weight_decay=0.01 \
    \
    training.ema=0.9999 \
    \
    wandb.project=continuous-embedding-diffusion \
    wandb.name="ced-owt-stage1"
