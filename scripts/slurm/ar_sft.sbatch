#!/usr/bin/env bash
#SBATCH --job-name=bqwen-ar-sft
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Paper cell C3: matched AR SFT (causal) on Nemotron, same budget as ar2block.
# Paper scale: submit with --nodes=8 (see submit_paper_cell.sh) so GBS=256 fits
# in Booster 12h wall.
#
# DDP: same as block_qwen — one Slurm task/GPU, trainer.devices=GPUs/node,
# CUDA_VISIBLE_DEVICES=0..N-1 (Slurm prolog single-GPU CVD breaks devices=4).
#
#   NUM_NODES=8 sbatch --nodes=8 … scripts/slurm/ar_sft.sbatch
#   or: ./scripts/submit_paper_cell.sh C3

set -euo pipefail

export ASMAA_WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
export NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_GPUS="${NUM_GPUS:-$((NUM_NODES * GPUS_PER_NODE))}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="${ASMAA_WORKSPACE}/Diffusion-new/outputs/block_qwen/ar_sft_${SLURM_JOB_ID}"
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash
# shellcheck disable=SC1091
source scripts/_block_qwen_ddp.bash
resolve_block_qwen_ddp_resources

DATA_CACHE="${DATA_CACHE:-${ASMAA_WORKSPACE}/.cache/discrete_diffusion/block_qwen_sft_nemotron}"
WANDB_PROJECT="${WANDB_PROJECT:-block_qwen}"
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/hydra" "${DATA_CACHE}" slurm_logs

# Matched conversion math mix (same as C0 / C2-math / U0).
export NEMOTRON_SFT_SPLITS="${NEMOTRON_SFT_SPLITS:-chat,safety,science,math,code}"
export NEMOTRON_SFT_MAX_PER_SPLIT="${NEMOTRON_SFT_MAX_PER_SPLIT:-math=1000000,code=500000}"

HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-}"
_append() {
  local key="$1" value="$2"
  if [[ "${HYDRA_OVERRIDES}" != *"${key}="* ]]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES:+${HYDRA_OVERRIDES} }${key}=${value}"
  fi
}
_append "trainer.max_steps" "${MAX_STEPS:-6000}"
_append "model.length" "${SEQ_LEN:-2048}"
_append "loader.global_batch_size" "${GBS:-256}"
_append "model.forward_mode" "causal"
_append "model.load_pretrained" "true"
append_block_qwen_trainer_overrides

WANDB_RUN_ID="${WANDB_RUN_ID:-ar_sft_${SLURM_JOB_ID}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ar_sft_${SLURM_JOB_ID}}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

# Shared srun helper expects these names.
EXPERIMENT=ar_sft_qwen
ALGO=ar_sft
export EXPERIMENT ALGO DATA_CACHE RUN_ROOT WANDB_PROJECT WANDB_RUN_NAME WANDB_RUN_ID WANDB_RESUME
export RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-true}"

echo "=== ar_sft launch (C3) ==="
echo "  run_root: ${RUN_ROOT}"
echo "  num_nodes: ${NUM_NODES}"
echo "  gpus/node: ${GPUS_PER_NODE}"
echo "  num_gpus: ${NUM_GPUS}"
echo "  cvd:      $(_cuda_visible_all_node_gpus)"
echo "  splits:   ${NEMOTRON_SFT_SPLITS}"
echo "  caps:     ${NEMOTRON_SFT_MAX_PER_SPLIT}"
echo "  wandb:    ${WANDB_PROJECT}/${WANDB_RUN_ID}"
echo "  overrides:${HYDRA_OVERRIDES}"

EXTRA_OVERRIDES=()
if [[ -n "${HYDRA_OVERRIDES}" ]]; then
  mapfile -t EXTRA_OVERRIDES < <(
    python - "${HYDRA_OVERRIDES}" <<'PY'
import shlex
import sys
for token in shlex.split(sys.argv[1]):
  print(token)
PY
  )
fi

TRAIN_RC=0
# EXTRA_OVERRIDES is a bash array in this shell; run_block_qwen_srun_train expands it.
run_block_qwen_srun_train || TRAIN_RC=$?

# shellcheck disable=SC1091
source scripts/_block_qwen_ckpt.bash
CKPT_DIR="${RUN_ROOT}/checkpoints"
MAX_STEPS_DONE="$(_block_qwen_max_steps)"
CKPT=""
STEP=-1
if PICK="$(_block_qwen_pick_highest_ckpt "${CKPT_DIR}" 2>/dev/null)"; then
  CKPT="${PICK%%$'\t'*}"
  STEP="${PICK#*$'\t'}"
fi

if [[ -n "${CKPT}" && "${STEP}" =~ ^[0-9]+$ && "${STEP}" -ge "${MAX_STEPS_DONE}" ]]; then
  if [[ "${RUN_FULL_EVAL:-true}" == "true" ]]; then
    echo "=== AR SFT finished at step ${STEP}; submitting lm_eval for ${CKPT} ==="
    export CKPT
    # Causal AR — standard lm-eval path (not block family router).
    env -u TMPDIR -u TEMP -u TMP \
      sbatch --export=ALL,CKPT,TASKS,NUM_STEPS,MAX_NEW_TOKENS,SKIP_THROUGHPUT,OUT_DIR,SUITE,NUM_NODES,GPUS_PER_NODE \
      --nodes="${NUM_NODES:-8}" --ntasks-per-node="${GPUS_PER_NODE:-4}" --gres=gpu:"${GPUS_PER_NODE:-4}" \
      --time=12:00:00 --job-name=bqwen-lm-eval \
      scripts/slurm/lm_eval.sbatch \
      || echo "WARNING: failed to submit lm_eval for ${CKPT}" >&2
  fi
elif [[ -n "${CKPT}" && "${STEP}" =~ ^[0-9]+$ && "${STEP}" -lt "${MAX_STEPS_DONE}" ]]; then
  echo "=== AR SFT incomplete: step ${STEP} < max_steps=${MAX_STEPS_DONE} (train_rc=${TRAIN_RC}) ==="
  echo "  Resume: NUM_NODES=${NUM_NODES} RUN_ROOT=${RUN_ROOT} sbatch --nodes=${NUM_NODES} scripts/slurm/ar_sft.sbatch" >&2
fi

exit "${TRAIN_RC}"
