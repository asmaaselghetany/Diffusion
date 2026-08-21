#!/usr/bin/env bash
# Scratch block uniform. Upstream-style Hydra + WandB recipe.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="src:${PYTHONPATH:-}"

WANDB_PROJECT="${WANDB_PROJECT:-UNI-D2}"
RUN_NAME="${WANDB_NAME:-block_uniform}"

python -u -m discrete_diffusion \
  +experiment=block_qwen \
  algo=block_uniform \
  model.load_pretrained=false \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${RUN_NAME}" \
  wandb.group=block_qwen_four_arms \
  'wandb.tags=[block_qwen,scratch,uniform]' \
  wandb.resume=allow \
  eval.t_bucketed_nll=true \
  checkpointing.resume_from_ckpt=false \
  hydra.run.dir="./outputs/block_qwen/${RUN_NAME}"
