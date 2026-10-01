#!/usr/bin/env bash
# Resume block_qwen training into an existing run directory.
#
# Guarantees:
#   1. RUN_ROOT is pinned to the existing dir (never a fresh job-id dir).
#   2. Resumes from the highest-step *valid* ckpt (zip-ok). No 24G Lustre
#      copy — passes checkpointing.resume_ckpt_path= that file directly.
#   3. Lightning resume_from_ckpt=true.
#   4. WandB continues the *same* cloud run (id/name from hydra overrides).
#   5. The original Hydra recipe and Slurm world size are replayed. Resource
#      migration is rejected by default because it changes batch semantics.
#   6. Walltime default 12h (Jupiter booster QOS cap). Override with TIME=.
#      Incomplete runs: re-run this script (or rely on AUTO_RESUME=1 from launch).
#
# Usage:
#   ./scripts/resume_block_qwen.sh outputs/block_qwen/ar2block_masked_139760
#   TIME=24:00:00 ./scripts/resume_block_qwen.sh outputs/block_qwen/block_masked_139762
#   DRY_RUN=1 ./scripts/resume_block_qwen.sh ...   # print plan, do not sbatch

set -euo pipefail

WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
_SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
_REQUESTED_REPO_ROOT="${RESUME_REPO_ROOT:-${_SCRIPT_REPO_ROOT}}"
_REQUESTED_SCRATCH="${SCRATCH:-}"
_REQUESTED_VENV_ROOT="${DIFFUSION_VENV_ROOT:-}"
_REQUESTED_HF_HOME="${HF_HOME:-}"
_REQUESTED_HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"
_REQUESTED_DIFFUSION_CACHE="${DISCRETE_DIFFUSION_SCRATCH_DIR:-}"
_REQUESTED_DATA_CACHE="${DATA_CACHE:-}"
_REQUESTED_HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-}"
_REQUESTED_NUM_NODES="${NUM_NODES:-}"
_REQUESTED_GPUS_PER_NODE="${GPUS_PER_NODE:-}"
_REQUESTED_WANDB_MODE="${WANDB_MODE:-}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
if [[ -n "${_REQUESTED_REPO_ROOT}" ]]; then
  REPO_ROOT="${_REQUESTED_REPO_ROOT}"
  export REPO_ROOT
fi
if [[ -n "${_REQUESTED_SCRATCH}" ]]; then export SCRATCH="${_REQUESTED_SCRATCH}"; fi
if [[ -n "${_REQUESTED_VENV_ROOT}" ]]; then export DIFFUSION_VENV_ROOT="${_REQUESTED_VENV_ROOT}"; fi
if [[ -n "${_REQUESTED_HF_HOME}" ]]; then export HF_HOME="${_REQUESTED_HF_HOME}"; fi
if [[ -n "${_REQUESTED_HF_DATASETS_CACHE}" ]]; then export HF_DATASETS_CACHE="${_REQUESTED_HF_DATASETS_CACHE}"; fi
if [[ -n "${_REQUESTED_DIFFUSION_CACHE}" ]]; then export DISCRETE_DIFFUSION_SCRATCH_DIR="${_REQUESTED_DIFFUSION_CACHE}"; fi
if [[ -n "${_REQUESTED_DATA_CACHE}" ]]; then export DATA_CACHE="${_REQUESTED_DATA_CACHE}"; fi
DIFFUSION_VENV_ROOT="${DIFFUSION_VENV_ROOT:-${WORKSPACE}/Diffusion}"
export DIFFUSION_VENV_ROOT
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source scripts/jupiter_paths.sh
# shellcheck disable=SC1091
source scripts/_block_qwen_ckpt.bash

# The checkpoint helpers invoke ``python`` before the submitted job sources
# _common.sh. Login shells on Jupiter do not necessarily expose a system
# Python, so make the requested training venv explicit here as well.
VENV_PYTHON="${DIFFUSION_VENV_ROOT}/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing training venv Python: ${VENV_PYTHON}" >&2
  exit 1
fi
export PATH="$(dirname "${VENV_PYTHON}"):${PATH}"

RUN_DIR_INPUT="${1:?Usage: resume_block_qwen.sh <run_dir>}"
if [[ ! -d "${RUN_DIR_INPUT}" ]]; then
  echo "Run directory not found: ${RUN_DIR_INPUT}" >&2
  exit 1
fi

RUN_DIR="$(readlink -f "${RUN_DIR_INPUT}")"
CKPT_DIR="${RUN_DIR}/checkpoints"
HYDRA_OVERRIDES_FILE="${RUN_DIR}/hydra/.hydra/overrides.yaml"
HYDRA_CONFIG_FILE="${RUN_DIR}/hydra/.hydra/config.yaml"
TIME_LIMIT="${TIME:-12:00:00}"

for metadata in "${HYDRA_OVERRIDES_FILE}" "${HYDRA_CONFIG_FILE}"; do
  if [[ ! -f "${metadata}" ]]; then
    echo "ERROR: cannot faithfully resume without recorded Hydra metadata: ${metadata}" >&2
    exit 1
  fi
done

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
  line="$(awk -v prefix="- ${key}=" \
    'index($0, prefix) == 1 {line=$0} END {print line}' \
    "${HYDRA_OVERRIDES_FILE}" 2>/dev/null || true)"
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

# Replay every original recipe override except launch-owned identity, paths,
# and resource keys. The caller's requested overrides are merged last by key.
HYDRA_OVERRIDES="$("${VENV_PYTHON}" - \
  "${HYDRA_OVERRIDES_FILE}" "${_REQUESTED_HYDRA_OVERRIDES}" <<'PY'
import shlex
import sys
from omegaconf import OmegaConf

path, requested = sys.argv[1:]
original = []
if path:
  try:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if isinstance(loaded, list):
      original = [str(item) for item in loaded]
  except FileNotFoundError:
    pass

owned = {
    '+experiment', 'experiment', 'algo',
    # Launch re-pins sampling=block|block_uniform per arm; never replay an
    # old wrong group (e.g. uniform trained under sampling=block).
    'sampling',
    'checkpointing.save_dir', 'checkpointing.resume_from_ckpt',
    'checkpointing.resume_ckpt_path', 'hydra.run.dir',
    'trainer.devices', 'trainer.num_nodes', 'strategy',
    'loader.eval_global_batch_size',
    'data.cache_dir',
}

def key(token):
  raw = token.split('=', 1)[0]
  return raw.lstrip('+~')

def launch_owned(token):
  raw = token.split('=', 1)[0]
  canonical = key(token)
  return raw in owned or canonical in owned or canonical.startswith('wandb.')

merged = {}
for token in original:
  if '=' in token and not launch_owned(token):
    merged[key(token)] = token
for token in shlex.split(requested):
  if '=' in token and not launch_owned(token):
    merged[key(token)] = token
print(shlex.join(merged.values()))
PY
)"

ORIG_NUM_NODES="$(_read_override trainer.num_nodes)"
ORIG_GPUS_PER_NODE="$(_read_override trainer.devices)"
ORIG_DATA_CACHE="$(_read_override data.cache_dir)"
if [[ -z "${ORIG_NUM_NODES}" || -z "${ORIG_GPUS_PER_NODE}" ]]; then
  echo "ERROR: recorded trainer.num_nodes/devices are required for a faithful resume" >&2
  exit 1
fi
NUM_NODES="${_REQUESTED_NUM_NODES:-${ORIG_NUM_NODES}}"
GPUS_PER_NODE="${_REQUESTED_GPUS_PER_NODE:-${ORIG_GPUS_PER_NODE}}"
if [[ "${ALLOW_RESOURCE_MIGRATION:-0}" != "1" ]] && \
   { [[ "${NUM_NODES}" != "${ORIG_NUM_NODES}" ]] || [[ "${GPUS_PER_NODE}" != "${ORIG_GPUS_PER_NODE}" ]]; }; then
  echo "ERROR: requested ${NUM_NODES}x${GPUS_PER_NODE} differs from recorded ${ORIG_NUM_NODES}x${ORIG_GPUS_PER_NODE}" >&2
  echo "Set ALLOW_RESOURCE_MIGRATION=1 only after validating global batch size and optimizer semantics." >&2
  exit 1
fi
NUM_GPUS="$((NUM_NODES * GPUS_PER_NODE))"
DATA_CACHE="${_REQUESTED_DATA_CACHE:-${ORIG_DATA_CACHE}}"
if [[ -z "${DATA_CACHE}" ]]; then
  DATA_CACHE="${DISCRETE_DIFFUSION_SCRATCH_DIR}/block_qwen_sft_nemotron"
fi

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

HYDRA_OVERRIDES="${HYDRA_OVERRIDES:+${HYDRA_OVERRIDES} }checkpointing.resume_ckpt_path=${CKPT}"

export RUN_ROOT="${RUN_DIR}"
export LINE="${LINE}"
export RESUME_FROM_CKPT=true
# Compute nodes cannot reach api.wandb.ai; force offline (sync later).
export WANDB_FORCE_OFFLINE=1
export WANDB_MODE=offline
export AUTO_RESUME="${AUTO_RESUME:-0}"
export RUN_FULL_EVAL="${RUN_FULL_EVAL:-false}"
export WANDB_PROJECT
export WANDB_RUN_ID
export WANDB_RUN_NAME
export WANDB_RESUME
export HYDRA_OVERRIDES
export NUM_NODES
export GPUS_PER_NODE
export NUM_GPUS
export DATA_CACHE

echo "=== resume plan ==="
echo "  line/arm:     ${LINE} / ${ARM}"
echo "  run_dir:      ${RUN_ROOT}"
echo "  resume_ckpt:  ${CKPT}"
echo "  global_step:  ${STEP}  →  max_steps=${MAX_STEPS}  (remain $((MAX_STEPS - STEP)))"
echo "  sbatch:       ${SBATCH_SCRIPT}"
echo "  time:         ${TIME_LIMIT}"
echo "  resources:    ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs (${NUM_GPUS} total)"
echo "  data_cache:   ${DATA_CACHE}"
echo "  wandb:        project=${WANDB_PROJECT} id=${WANDB_RUN_ID} resume=${WANDB_RESUME}"
echo "  hydra replay: ${HYDRA_OVERRIDES}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — not submitting."
  exit 0
fi

# Strip eval knobs before --export=ALL (same class of bug as submit_lever).
unset FORCE_STACK FORCE_DECODE_PROFILE FORCE_UNMASK_THRESHOLD \
  FORCE_GREEDY_PIN FORCE_GREEDY FORCE_ARPC \
  OUT_DIR SUITE TASKS JOB_NAME DECODE_PROFILE UNMASK_THRESHOLD \
  ALLOW_FULL_SEQ_DECODE EVAL_DECODE_PROFILE ARPC_OUT HYGIENE_OUT CKPT \
  || true

JOB_ID="$(sbatch --parsable \
  --account="${JUWELS_ACCOUNT:-profound}" \
  --time="${TIME_LIMIT}" \
  --nodes="${NUM_NODES}" \
  --ntasks-per-node="${GPUS_PER_NODE}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --job-name="resume_${LINE}_${ARM}" \
  --chdir="${REPO_ROOT}" \
  --export=ALL,RUN_ROOT,LINE,RESUME_FROM_CKPT,WANDB_FORCE_OFFLINE,WANDB_MODE,WANDB_PROJECT,WANDB_RUN_ID,WANDB_RUN_NAME,WANDB_RESUME,HYDRA_OVERRIDES,NUM_NODES,GPUS_PER_NODE,NUM_GPUS,DATA_CACHE,AUTO_RESUME,RUN_FULL_EVAL \
  "${SBATCH_SCRIPT}")"
echo "Submitted ${JOB_ID}"
echo "${JOB_ID}"
