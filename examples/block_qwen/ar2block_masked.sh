#!/usr/bin/env bash
# AR→block masked (Fast-dLLM-style init). Upstream-style Hydra + WandB recipe.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="src:${PYTHONPATH:-}"

WANDB_PROJECT="${WANDB_PROJECT:-UNI-D2}"
RUN_NAME="${WANDB_NAME:-ar2block_masked}"

python -u -m discrete_diffusion \
  +experiment=block_qwen \
  algo=block_masked \
  sampling=block \
  model.load_pretrained=true \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${RUN_NAME}" \
  wandb.group=block_qwen_four_arms \
  'wandb.tags=[block_qwen,ar2block,masked]' \
  wandb.resume=allow \
  eval.t_bucketed_nll=true \
  checkpointing.resume_from_ckpt=false \
  hydra.run.dir="./outputs/block_qwen/${RUN_NAME}"
