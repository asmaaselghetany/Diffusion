#!/usr/bin/env bash
# Launch one Qwen block arm (block_qwen experiment).
#
# LINE (required via env, or inferred from RUN_ROOT basename):
#   ar2block  — Pipeline 1: AR→block (from_pretrained Qwen)
#   block     — Pipeline 2: pure block diffusion (scratch init)
#   blockgen  — legacy alias for block (old run dirs blockgen_*)
#
# Usage (from repo root, after sourcing _block_qwen_env.bash):
#   LINE=ar2block scripts/_block_qwen_launch.bash masked
#   LINE=block scripts/_block_qwen_launch.bash uniform
#   LINE=ar2block scripts/_block_qwen_launch.bash hybrid

set -euo pipefail

ARM="${1:?Usage: _block_qwen_launch.bash <masked|uniform|hybrid>}"

case "${ARM}" in
  masked)  ALGO=block_masked ;;
  uniform) ALGO=block_uniform ;;
  hybrid)  ALGO=block_hybrid ;;
  *)
    echo "Unknown arm: ${ARM} (expected masked, uniform, or hybrid)" >&2
    exit 1
    ;;
esac

# Infer LINE from RUN_ROOT if already pinned (resume), else require LINE.
if [[ -z "${LINE:-}" && -n "${RUN_ROOT:-}" ]]; then
  _base="$(basename "${RUN_ROOT}")"
  if [[ "${_base}" =~ ^(ar2block|block|blockgen)_ ]]; then
    LINE="${BASH_REMATCH[1]}"
  elif [[ "${_base}" =~ ^(masked|uniform)_ ]]; then
    # Legacy neutral dirs are Pipeline 1 (AR init).
    LINE="ar2block"
  fi
fi

LINE="${LINE:-ar2block}"
case "${LINE}" in
  ar2block|block|blockgen) ;;
  *)
    echo "Unknown LINE=${LINE} (expected ar2block|block|blockgen)" >&2
    exit 1
    ;;
esac

EXPERIMENT=block_qwen
DATA_CACHE="${DATA_CACHE:-${ASMAA_WORKSPACE}/.cache/discrete_diffusion/block_qwen_sft_nemotron}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/block_qwen/${LINE}_${ARM}_${SLURM_JOB_ID:-local}}"
# Paper curves live in block_qwen. Track 1/2 micros set WANDB_PROJECT=block_qwen_trials.
WANDB_PROJECT="${WANDB_PROJECT:-block_qwen}"
NUM_GPUS="${NUM_GPUS:-2}"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/hydra" "${DATA_CACHE}" slurm_logs
echo "Dataset cache at ${DATA_CACHE} (built on first train epoch)."

_append_override() {
  local key="$1"
  local value="$2"
  if [[ "${HYDRA_OVERRIDES:-}" != *"${key}="* ]]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES:+${HYDRA_OVERRIDES} }${key}=${value}"
  fi
}

# Multi-GPU under Slurm (Lightning-compatible):
#   --ntasks-per-node=NUM_GPUS  (NOT plain --ntasks)
#   --gres=gpu:NUM_GPUS
#   trainer.devices=NUM_GPUS
# Do NOT pass --gpus-per-task=1: that remaps each task to only GPU [0], which
# breaks devices=NUM_GPUS. Lightning binds ranks via LOCAL_RANK instead.
_append_override "trainer.devices" "${NUM_GPUS}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  _append_override "strategy" "ddp"
fi
# Fair paired compare: same validation batch on both arms (= NUM_GPUS).
_append_override "loader.eval_global_batch_size" "${NUM_GPUS}"
# Keep yaml log_every_n_steps (50); do not force 10 — sync/wandb overhead.
# Pipeline 2: random init (architecture from hub config only).
if [[ "${LINE}" == "block" || "${LINE}" == "blockgen" ]]; then
  _append_override "model.load_pretrained" "false"
fi

RUN_BASENAME="$(basename "${RUN_ROOT}")"
# Canonical 6000-step paper arms → stable ids in project block_qwen.
# Everything else (Track 1/2 micros, ad-hoc) → job-unique id, never resume/append.
_CANONICAL_ROOTS=(
  ar2block_masked_131655
  ar2block_uniform_133161
  block_masked_133150
  block_uniform_133151
)
_is_canonical=0
for _root in "${_CANONICAL_ROOTS[@]}"; do
  if [[ "${RUN_BASENAME}" == "${_root}" ]]; then
    _is_canonical=1
    break
  fi
done
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  : # explicit override wins (callers must pass a unique id + WANDB_RESUME)
elif [[ "${_is_canonical}" -eq 1 ]]; then
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-${LINE}_${ARM}}"
  WANDB_RUN_ID="${LINE}_${ARM}_v9"
  WANDB_RESUME="${WANDB_RESUME:-allow}"
else
  _wid_suffix="${SLURM_JOB_ID:-$$}"
  # Always suffix job id so names never collide in the trials UI.
  _base_name="${WANDB_RUN_NAME:-${LINE}_${ARM}}"
  WANDB_RUN_NAME="${_base_name}_${_wid_suffix}"
  WANDB_RUN_ID="${_base_name}_${_wid_suffix}"
  # never append into an existing cloud run — collision must fail loudly
  WANDB_RESUME="${WANDB_RESUME:-never}"
fi
WANDB_RESUME="${WANDB_RESUME:-allow}"
echo "=== block_qwen launch ==="
echo "  line:       ${LINE}"
echo "  arm:        ${ARM} (${ALGO})"
echo "  experiment: ${EXPERIMENT}"
echo "  run_root:   ${RUN_ROOT}"
echo "  data_cache: ${DATA_CACHE}"
echo "  num_gpus:   ${NUM_GPUS}"
echo "  wandb_project: ${WANDB_PROJECT}"
echo "  wandb_name: ${WANDB_RUN_NAME}"
echo "  wandb_id:   ${WANDB_RUN_ID}"
echo "  wandb_resume: ${WANDB_RESUME}"
echo "  wandb_mode: ${WANDB_MODE:-unset} (key=${WANDB_API_KEY:+set})"
if [[ "${LINE}" == "block" || "${LINE}" == "blockgen" ]]; then
  echo "  init:       scratch (model.load_pretrained=false)"
else
  echo "  init:       AR pretrained (model.load_pretrained=true)"
fi

EXTRA_OVERRIDES=()
if [[ -n "${HYDRA_OVERRIDES:-}" ]]; then
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
# One Slurm task per GPU so Lightning DDP sees world_size=NUM_GPUS
# (ntasks=1 + devices=2 → MEMBER 1/1 and a wasted GPU).
srun --ntasks-per-node="${NUM_GPUS}" --cpu-bind=cores \
  python -u -m discrete_diffusion "+experiment=${EXPERIMENT}" "algo=${ALGO}" \
  data.cache_dir="${DATA_CACHE}" \
  checkpointing.save_dir="${RUN_ROOT}" \
  checkpointing.resume_from_ckpt="${RESUME_FROM_CKPT:-true}" \
  hydra.run.dir="${RUN_ROOT}/hydra" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${WANDB_RUN_NAME}" \
  "wandb.id=${WANDB_RUN_ID}" \
  "wandb.resume=${WANDB_RESUME}" \
  "${EXTRA_OVERRIDES[@]}" || TRAIN_RC=$?

# After training: eval if highest valid ckpt reached max_steps; otherwise
# auto-resubmit resume (partition MaxTime is 24h). Set AUTO_RESUME=0 to disable.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_block_qwen_ckpt.bash"
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
    echo "=== Training finished at step ${STEP} (>= max_steps=${MAX_STEPS}); submitting eval for ${CKPT} ==="
    sbatch scripts/slurm/eval_checkpoint.sbatch "${CKPT}" \
      || echo "WARNING: failed to submit eval job for ${CKPT}" >&2
  fi
elif [[ -n "${CKPT}" && "${STEP}" =~ ^[0-9]+$ && "${STEP}" -lt "${MAX_STEPS}" ]]; then
  echo "=== Incomplete: step ${STEP} < max_steps=${MAX_STEPS} (train_rc=${TRAIN_RC}) ==="
  if [[ "${AUTO_RESUME:-1}" == "1" ]]; then
    echo "AUTO_RESUME=1 → submitting next chunk into ${RUN_ROOT}"
    # Preserve sample/wandb env; resume script re-pins RUN_ROOT + highest ckpt.
    "${REPO_ROOT}/scripts/resume_block_qwen.sh" "${RUN_ROOT}" \
      || echo "WARNING: auto-resume submit failed" >&2
  else
    echo "  Resume manually: ./scripts/resume_block_qwen.sh ${RUN_ROOT}" >&2
  fi
elif [[ "${TRAIN_RC}" -ne 0 ]]; then
  echo "Training exited ${TRAIN_RC}; no valid ckpt under ${CKPT_DIR}."
  echo "  Resume: ./scripts/resume_block_qwen.sh ${RUN_ROOT}"
else
  echo "WARNING: no valid checkpoint under ${CKPT_DIR}; skipping eval/resume" >&2
fi

exit "${TRAIN_RC}"
