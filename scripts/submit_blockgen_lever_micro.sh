#!/usr/bin/env bash
# BlockGen-derived *shared* lever micros (hooks off). Draft only — do NOT
# launch until Track 1 1500 (138143/138144) returns and you pick one letter.
#
# Single-factor only. Do not bundle mixture + t-strat in one micro.
#
# Usage (after decision):
#   LEVER=A ./scripts/submit_blockgen_lever_micro.sh both
#   LEVER=B MAX_STEPS=500 ./scripts/submit_blockgen_lever_micro.sh both
#
# LEVER=A: block_size_mixture=[16,32]     — BlockGen multi-size analogue
# LEVER=B: stratified_gamma=0.5           — OUR t-strat (≠ BlockGen u-stratified)
# LEVER=C: block_size_mixture=[1,32]      — AR-size component; ARPC prereq
# LEVER=D: C + sampling.use_arpc=true     — decode; only after C is understood
#
# Not a lever: loss_type=elbo — already the uniform (and masked) default.
# Not implemented: BlockGen block_size_per_gpu=u-stratified (weighted 2^k draw).

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

MAX_STEPS="${MAX_STEPS:-500}"
SEQ_LEN="${SEQ_LEN:-512}"
BLOCK="${BLOCK:-32}"
GBS="${GBS:-128}"
LEVER="${LEVER:-A}"

case "${LEVER}" in
  A)
    EXTRA="algo.block_size_mixture=[16,32]"
    TAG="bgA_mix16_32"
    ;;
  B)
    EXTRA="algo.stratified_gamma=0.5"
    TAG="bgB_tstrat05"
    ;;
  C)
    EXTRA="algo.block_size_mixture=[1,32]"
    TAG="bgC_mix1_32"
    ;;
  D)
    EXTRA="algo.block_size_mixture=[1,32] sampling.use_arpc=true"
    TAG="bgD_mix1_32_arpc"
    echo "WARN: LEVER=D is decode+train mixture; run C first." >&2
    ;;
  *)
    echo "LEVER must be A|B|C|D (got ${LEVER})" >&2
    exit 1
    ;;
esac

export RESUME_FROM_CKPT=false
export RUN_FULL_EVAL=false
export WANDB_PROJECT="${WANDB_PROJECT:-block_qwen_trials}"
export WANDB_RESUME=never
export HYDRA_OVERRIDES="trainer.max_steps=${MAX_STEPS} model.length=${SEQ_LEN} block_size=${BLOCK} loader.global_batch_size=${GBS} eval.t_bucketed_nll=true ${EXTRA}"

echo "BlockGen lever micro LEVER=${LEVER} TAG=${TAG}"
echo "HYDRA_OVERRIDES=${HYDRA_OVERRIDES}"
echo "See docs/BLOCKGEN_LEVERS.md — directional launch OK; tax needs harness-PASS; promote-to-default needs AND."

submit() {
  local arm="$1"
  export WANDB_RUN_NAME="blockgen_lever_${TAG}_${arm}"
  unset WANDB_RUN_ID || true
  sbatch --job-name="bg_lever_${TAG}_${arm}" \
    --export=ALL,WANDB_PROJECT,WANDB_RESUME,WANDB_RUN_NAME,RESUME_FROM_CKPT,RUN_FULL_EVAL,HYDRA_OVERRIDES \
    "scripts/slurm/ar2block_${arm}.sbatch"
}

TARGET="${1:-both}"
case "${TARGET}" in
  masked|uniform) submit "${TARGET}" ;;
  both|all) submit masked; submit uniform ;;
  *)
    echo "Usage: LEVER=A|B|C|D \$0 {masked|uniform|both}" >&2
    exit 1
    ;;
esac
