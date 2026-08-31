#!/usr/bin/env bash
# Resume block_qwen training into an existing run directory.
#
# Guarantees:
#   1. RUN_ROOT is pinned to the existing dir (never a fresh job-id dir).
#   2. Resumes from the highest-step *valid* ckpt (zip-ok). No 24G Lustre
#      copy — passes checkpointing.resume_ckpt_path= that file directly.
#   3. Lightning resume_from_ckpt=true.
#   4. WandB continues the *same* cloud run (id/name from hydra overrides).
#   5. Sample / t-bucket hydra overrides are re-applied if the original run had them.
#   6. Walltime default 24h (HAICORE ``standard`` MaxTime). Override with TIME=.
#      Incomplete runs: re-run this script (or rely on AUTO_RESUME=1 from launch).
#
# Usage:
#   ./scripts/resume_block_qwen.sh outputs/block_qwen/ar2block_masked_139760
#   TIME=24:00:00 ./scripts/resume_block_qwen.sh outputs/block_qwen/block_masked_139762
#   DRY_RUN=1 ./scripts/resume_block_qwen.sh ...   # print plan, do not sbatch

set -euo pipefail

WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source scripts/_block_qwen_ckpt.bash

RUN_DIR_INPUT="${1:?Usage: resume_block_qwen.sh <run_dir>}"
if [[ ! -d "${RUN_DIR_INPUT}" ]]; then
  echo "Run directory not found: ${RUN_DIR_INPUT}" >&2
  exit 1
fi

RUN_DIR="$(readlink -f "${RUN_DIR_INPUT}")"
CKPT_DIR="${RUN_DIR}/checkpoints"
HYDRA_OVERRIDES_FILE="${RUN_DIR}/hydra/.hydra/overrides.yaml"
TIME_LIMIT="${TIME:-24:00:00}"

_block_qwen_require_disk "${RUN_DIR}" || exit 1

# shellcheck disable=SC1091
source scripts/_resolve_block_qwen_run.bash
read -r LINE ARM ORIG_JOB_ID < <(parse_block_qwen_run_basename "$(basename "${RUN_DIR}")")

BASE="$(basename "${RUN_DIR}")"
if [[ "${BASE}" =~ ^(masked|uniform)_([0-9]+)$ ]]; then
  SBATCH_SCRIPT="scripts/slurm/${ARM}.sbatch"
else
  SBATCH_SCRIPT="scripts/slurm/${LINE}_${ARM}.sbatch"
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "Missing sbatch script: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

_read_override() {
  local key="$1"
  [[ -f "${HYDRA_OVERRIDES_FILE}" ]] || return 0
  local line
  line="$(grep -E "^- ${key}=" "${HYDRA_OVERRIDES_FILE}" 2>/dev/null | tail -1 || true)"
  if [[ -n "${line}" ]]; then
    echo "${line#*=}"
  fi
}

WANDB_PROJECT="${WANDB_PROJECT:-$(_read_override wandb.project)}"
WANDB_PROJECT="${WANDB_PROJECT:-block_qwen_trials}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$(_read_override wandb.id)}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(_read_override wandb.name)}"
if [[ -z "${WANDB_RUN_ID}" ]]; then
  WANDB_RUN_ID="clean_graph_${LINE}_${ARM}_${ORIG_JOB_ID}"
  WANDB_RUN_NAME="${WANDB_RUN_ID}"
fi
WANDB_RESUME="${WANDB_RESUME:-allow}"

if [[ -z "${HYDRA_OVERRIDES:-}" && -f "${HYDRA_OVERRIDES_FILE}" ]]; then
  HYDRA_OVERRIDES=""
  while IFS= read -r line; do
    [[ "${line}" =~ ^-[[:space:]]+(.*)$ ]] || continue
    kv="${BASH_REMATCH[1]}"
    key="${kv%%=*}"
    case "${key}" in
      checkpointing.resume_ckpt_path|checkpointing.save_dir) continue ;;
    esac
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES:+${HYDRA_OVERRIDES} }${kv}"
  done < "${HYDRA_OVERRIDES_FILE}"
fi

# Pick highest valid ckpt; do NOT cp onto last.ckpt (avoids multi-GB Lustre copy
# when last was truncated mid-timeout). Point Lightning at the file directly.
PICK="$(_block_qwen_pick_highest_ckpt "${CKPT_DIR}")" || {
  echo "ERROR: no valid checkpoint in ${CKPT_DIR}" >&2
  exit 1
}
CKPT="${PICK%%$'\t'*}"
STEP="${PICK#*$'\t'}"
MAX_STEPS="$(_block_qwen_max_steps)"

if [[ ! "${STEP}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not resolve global_step for ${CKPT} (got '${STEP}')" >&2
  exit 1
fi
if [[ "${STEP}" -ge "${MAX_STEPS}" ]]; then
  echo "Already at step ${STEP} >= max_steps=${MAX_STEPS}; nothing to resume."
  echo "  ckpt: ${CKPT}"
  exit 0
fi

# Force Lightning to load this exact file (may be 0-3500.ckpt, not last.ckpt).
HYDRA_OVERRIDES="${HYDRA_OVERRIDES:+${HYDRA_OVERRIDES} }checkpointing.resume_ckpt_path=${CKPT}"

export RUN_ROOT="${RUN_DIR}"
export LINE="${LINE}"
export RESUME_FROM_CKPT=true
export WANDB_MODE=online
export WANDB_PROJECT
export WANDB_RUN_ID
export WANDB_RUN_NAME
export WANDB_RESUME
export HYDRA_OVERRIDES

echo "=== resume plan ==="
echo "  line/arm:     ${LINE} / ${ARM}"
echo "  run_dir:      ${RUN_ROOT}"
echo "  resume_ckpt:  ${CKPT}"
echo "  global_step:  ${STEP}  →  max_steps=${MAX_STEPS}  (remain $((MAX_STEPS - STEP)))"
echo "  sbatch:       ${SBATCH_SCRIPT}"
echo "  time:         ${TIME_LIMIT}"
echo "  wandb:        project=${WANDB_PROJECT} id=${WANDB_RUN_ID} resume=${WANDB_RESUME}"
echo "  hydra extras: ${HYDRA_OVERRIDES}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — not submitting."
  exit 0
fi

JOB_ID="$(sbatch --parsable \
  --time="${TIME_LIMIT}" \
  --job-name="resume_${LINE}_${ARM}" \
  --export=ALL,RUN_ROOT,LINE,RESUME_FROM_CKPT,WANDB_MODE,WANDB_PROJECT,WANDB_RUN_ID,WANDB_RUN_NAME,WANDB_RESUME,HYDRA_OVERRIDES \
  "${SBATCH_SCRIPT}")"
echo "Submitted ${JOB_ID}"
echo "${JOB_ID}"
