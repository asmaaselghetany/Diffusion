#!/usr/bin/env bash
# U0_from_C0 — gated two-stage AR→masked→uniform continue-FT.
#
# SAFETY:
#   - NEVER writes into the C0 run directory.
#   - Fresh RUN_ROOT: ar2block_uniform_<jobid> via U0 lever submit.
#   - Fresh WandB id (WANDB_RESUME=never).
#   - Loads weights from C0 ckpt only (checkpointing.resume_ckpt_path).
#
# Usage:
#   ./scripts/submit_u0_from_c0.sh
#   C0_RUN=.../ar2block_masked_1762534 MAX_STEPS=7500 ./scripts/submit_u0_from_c0.sh
#   DRY_RUN=1 ./scripts/submit_u0_from_c0.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
# Login shells may lack `python`; pin training venv before ckpt helpers.
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
VENV_PY="${DIFFUSION_VENV_ROOT:-${WORKSPACE}/Diffusion}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  VENV_PY="${REPO_ROOT}/.venv/bin/python"
fi
export PATH="$(dirname "${VENV_PY}"):${PATH}"
# shellcheck disable=SC1091
source scripts/_block_qwen_ckpt.bash

C0_RUN="${C0_RUN:-${REPO_ROOT}/outputs/block_qwen/ar2block_masked_1762534}"
# Prefer Diffusion-new; fall back to Diffusion mirror.
if [[ ! -d "${C0_RUN}/checkpoints" ]]; then
  ALT="/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_masked_1762534"
  if [[ -d "${ALT}/checkpoints" ]]; then
    C0_RUN="${ALT}"
  fi
fi
if [[ ! -d "${C0_RUN}/checkpoints" ]]; then
  echo "C0 checkpoints not found under ${C0_RUN}" >&2
  exit 1
fi

PICK=""
SRC_CKPT=""
SRC_STEP=""
if [[ -f "${C0_RUN}/checkpoints/last.ckpt" ]]; then
  SRC_CKPT="${C0_RUN}/checkpoints/last.ckpt"
  SRC_STEP="last"
else
  PICK="$(_block_qwen_pick_highest_ckpt "${C0_RUN}/checkpoints")"
  SRC_CKPT="${PICK%%$'\t'*}"
  SRC_STEP="${PICK#*$'\t'}"
fi
if [[ ! -f "${SRC_CKPT}" ]]; then
  echo "Failed to resolve C0 ckpt from ${C0_RUN}" >&2
  exit 1
fi

# Extra budget beyond C0's 6000 (continue-FT). Override with MAX_STEPS=.
export MAX_STEPS="${MAX_STEPS:-7500}"
export RESUME_FROM_CKPT=true
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-} checkpointing.resume_ckpt_path=${SRC_CKPT}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-U0_from_C0}"
export NEMOTRON_SFT_SPLITS="${NEMOTRON_SFT_SPLITS:-chat,safety,science,math,code}"
export NEMOTRON_SFT_MAX_PER_SPLIT="${NEMOTRON_SFT_MAX_PER_SPLIT:-math=1000000,code=500000}"
# Ensure we do not inherit a pinned RUN_ROOT that would pollute C0.
unset RUN_ROOT || true

echo "=== U0_from_C0 (isolated continue-FT) ==="
echo "  source_ckpt: ${SRC_CKPT} (step ${SRC_STEP})"
echo "  source_run:  ${C0_RUN}"
echo "  max_steps:   ${MAX_STEPS}"
echo "  wandb_name:  ${WANDB_RUN_NAME}"
echo "  NOTE: new job gets fresh ar2block_uniform_<jid>; C0 dir untouched"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1 ./scripts/submit_lever.sh --preset U0 --arm uniform --paper
else
  ./scripts/submit_lever.sh --preset U0 --arm uniform --paper
fi
