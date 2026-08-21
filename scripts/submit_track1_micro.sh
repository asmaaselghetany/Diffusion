#!/usr/bin/env bash
# Track 1 micro-runs: ar2block masked + uniform, neutral levers only.
# Smoke defaults: 500 steps, seq 512, block 32, hooks off, no auto DepBench eval.
#
# WandB: project block_qwen_trials (never block_qwen). Each Slurm job gets a
# unique run id; resume=never so nothing overwrites.
#
# Usage:
#   ./scripts/submit_track1_micro.sh          # both arms, 500-step smoke
#   MAX_STEPS=1500 ./scripts/submit_track1_micro.sh   # scale if smoke alive
#   ./scripts/submit_track1_micro.sh masked

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

MAX_STEPS="${MAX_STEPS:-500}"
SEQ_LEN="${SEQ_LEN:-512}"
BLOCK="${BLOCK:-32}"
# Keep global batch modest for smoke; bump is a single Track 1 factor later.
GBS="${GBS:-128}"

export RESUME_FROM_CKPT=false
export RUN_FULL_EVAL=false
# Trials project — isolated from paper block_qwen curves.
export WANDB_PROJECT="${WANDB_PROJECT:-block_qwen_trials}"
export WANDB_RESUME=never
# Layer-3: α-bucketed val NLL (AR-init cliff detector). Attention hook hygiene
# is in _block_qwen_env.bash (PYTHONPATH reset).
# generate_samples: one val table at max_steps (val_check_interval = 500 opt steps).
export HYDRA_OVERRIDES="trainer.max_steps=${MAX_STEPS} model.length=${SEQ_LEN} block_size=${BLOCK} loader.global_batch_size=${GBS} eval.t_bucketed_nll=true eval.generate_samples=true eval.save_validation_samples=true"

echo "Track 1 micro: steps=${MAX_STEPS} seq=${SEQ_LEN} block=${BLOCK} gbs=${GBS}"
echo "WANDB_PROJECT=${WANDB_PROJECT} (resume=${WANDB_RESUME})"
echo "HYDRA_OVERRIDES=${HYDRA_OVERRIDES}"

submit() {
  local arm="$1"
  local script="scripts/slurm/ar2block_${arm}.sbatch"
  # Name includes track so the trials UI is readable; id still job-unique in launch.
  export WANDB_RUN_NAME="track1_rope_${arm}"
  # Clear any stale id from a prior submit() in this shell; launch assigns job id.
  unset WANDB_RUN_ID || true
  sbatch --job-name="track1_micro_${arm}" \
    --export=ALL,WANDB_PROJECT,WANDB_RESUME,WANDB_RUN_NAME,RESUME_FROM_CKPT,RUN_FULL_EVAL,HYDRA_OVERRIDES \
    "${script}"
}

TARGET="${1:-both}"
case "${TARGET}" in
  masked|uniform) submit "${TARGET}" ;;
  both|all)
    submit masked
    submit uniform
    ;;
  *)
    echo "Usage: $0 {masked|uniform|both}" >&2
    exit 1
    ;;
esac

echo "After exit 0: run harness with --prompt-wrap sft on each run's prepared last.ckpt"
echo "Go: either arm pos VR < 0.5 and other Δ ≤ +0.05 vs baseline."
