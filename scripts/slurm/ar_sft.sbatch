#!/usr/bin/env bash
#SBATCH --job-name=bqwen-ar-sft
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --chdir=/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2
#
# Paper cell C3: matched AR SFT (causal) on Nemotron, same budget as ar2block.
#
# Submit: sbatch scripts/slurm/ar_sft.sbatch

set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-2}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2/outputs/block_qwen/ar_sft_${SLURM_JOB_ID}"
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

DATA_CACHE="${DATA_CACHE:-${ASMAA_WORKSPACE}/.cache/discrete_diffusion/block_qwen_sft_nemotron}"
WANDB_PROJECT="${WANDB_PROJECT:-block_qwen}"
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/hydra" "${DATA_CACHE}" slurm_logs

HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-}"
_append() {
  local key="$1" value="$2"
  if [[ "${HYDRA_OVERRIDES}" != *"${key}="* ]]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES:+${HYDRA_OVERRIDES} }${key}=${value}"
  fi
}
_append "trainer.devices" "${NUM_GPUS}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  _append "strategy" "ddp"
fi
_append "loader.eval_global_batch_size" "${NUM_GPUS}"

WANDB_RUN_ID="${WANDB_RUN_ID:-ar_sft_${SLURM_JOB_ID}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ar_sft_${SLURM_JOB_ID}}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

echo "=== ar_sft launch ==="
echo "  run_root: ${RUN_ROOT}"
echo "  num_gpus: ${NUM_GPUS}"
echo "  wandb:    ${WANDB_PROJECT}/${WANDB_RUN_ID}"
echo "  overrides:${HYDRA_OVERRIDES}"

# shellcheck disable=SC2086
srun --ntasks-per-node="${NUM_GPUS}" --cpu-bind=cores \
  python -u -m discrete_diffusion \
  "+experiment=ar_sft_qwen" \
  algo=ar_sft \
  model.forward_mode=causal \
  model.load_pretrained=true \
  checkpointing.save_dir="${RUN_ROOT}" \
  checkpointing.resume_from_ckpt="${RESUME_FROM_CKPT:-true}" \
  hydra.run.dir="${RUN_ROOT}/hydra" \
  data.cache_dir="${DATA_CACHE}" \
  loader.num_workers=4 \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${WANDB_RUN_NAME}" \
  wandb.id="${WANDB_RUN_ID}" \
  wandb.resume="${WANDB_RESUME}" \
  ${HYDRA_OVERRIDES}

TRAIN_RC=$?

# shellcheck disable=SC1091
source scripts/_block_qwen_ckpt.bash
CKPT_DIR="${RUN_ROOT}/checkpoints"
MAX_STEPS="$(_block_qwen_max_steps)"
CKPT=""
STEP=-1
if PICK="$(_block_qwen_pick_highest_ckpt "${CKPT_DIR}" 2>/dev/null)"; then
  CKPT="${PICK%%$'\t'*}"
  STEP="${PICK#*$'\t'}"
fi

if [[ -n "${CKPT}" && "${STEP}" =~ ^[0-9]+$ && "${STEP}" -ge "${MAX_STEPS}" ]]; then
  if [[ "${RUN_FULL_EVAL:-true}" == "true" ]]; then
    echo "=== AR SFT finished at step ${STEP}; submitting eval for ${CKPT} ==="
    sbatch scripts/slurm/eval_checkpoint.sbatch "${CKPT}" \
      --skip-elbo --skip-depbench \
      || echo "WARNING: failed to submit eval job for ${CKPT}" >&2
    export CKPT
    sbatch --export=ALL,CKPT,TASKS,NUM_STEPS,MAX_NEW_TOKENS,SKIP_THROUGHPUT,OUT_DIR \
      scripts/slurm/lm_eval.sbatch \
      || echo "WARNING: failed to submit lm_eval for ${CKPT}" >&2
    if [[ "${RUN_LONGITUDINAL:-false}" == "true" ]]; then
      RUN_ROOT="${RUN_ROOT}" sbatch --export=ALL,RUN_ROOT \
        --wrap="bash scripts/submit_longitudinal_eval.sh '${RUN_ROOT}'" \
        || echo "WARNING: failed to submit longitudinal eval" >&2
    fi
  fi
elif [[ -n "${CKPT}" && "${STEP}" =~ ^[0-9]+$ && "${STEP}" -lt "${MAX_STEPS}" ]]; then
  echo "=== AR SFT incomplete: step ${STEP} < max_steps=${MAX_STEPS} (train_rc=${TRAIN_RC}) ==="
  echo "  Resume manually with RESUME_FROM_CKPT=true and same RUN_ROOT" >&2
fi

exit "${TRAIN_RC}"
