#!/usr/bin/env bash
# Track 2 masked diagnostic micro: ar2block_masked with Fast-dLLM-style hooks.
# Informational only — never the shared masked↔uniform bakeoff recipe.
#
# WandB: project block_qwen_trials (never block_qwen). Unique job id; resume=never.
#
# Smoke: 500 steps, seq 512, block 32, shift + complementary, t_bucketed_nll.
#
# Usage:
#   ./scripts/submit_track2_masked_micro.sh
#   MAX_STEPS=1500 ./scripts/submit_track2_masked_micro.sh

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

MAX_STEPS="${MAX_STEPS:-500}"
SEQ_LEN="${SEQ_LEN:-512}"
BLOCK="${BLOCK:-32}"
GBS="${GBS:-128}"

export RESUME_FROM_CKPT=false
export RUN_FULL_EVAL=false
export WANDB_PROJECT="${WANDB_PROJECT:-block_qwen_trials}"
export WANDB_RESUME=never
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-track2_masked}"
unset WANDB_RUN_ID || true
# Shift + complementary (masked-native). t_bucketed_nll for Layer-3 curves.
# PYTHONPATH hygiene is in _block_qwen_env.bash.
export HYDRA_OVERRIDES="trainer.max_steps=${MAX_STEPS} model.length=${SEQ_LEN} block_size=${BLOCK} loader.global_batch_size=${GBS} algo.shift_loss_targets=true algo.complementary_masks=true eval.t_bucketed_nll=true"

echo "Track 2 masked micro: steps=${MAX_STEPS} seq=${SEQ_LEN} block=${BLOCK} gbs=${GBS}"
echo "WANDB_PROJECT=${WANDB_PROJECT} name=${WANDB_RUN_NAME} (resume=${WANDB_RESUME})"
echo "HYDRA_OVERRIDES=${HYDRA_OVERRIDES}"

sbatch --job-name="track2_micro_masked" \
  --export=ALL,WANDB_PROJECT,WANDB_RESUME,WANDB_RUN_NAME,RESUME_FROM_CKPT,RUN_FULL_EVAL,HYDRA_OVERRIDES \
  scripts/slurm/ar2block_masked.sbatch

echo "After exit 0: harness sft AND tools/run_block_arm_sanity.py on best.ckpt (conjunction)."
echo "Interpret: pos VR < 0.5 and low-t >= 0.90 => hooks can rescue masked; else deeper than missing hooks."
