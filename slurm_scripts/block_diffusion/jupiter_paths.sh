#!/usr/bin/env bash
# Jupiter (JSC Booster) path helpers — scifi group convention.
#
# Work only under /e/project1/scifi/... (visible on login nodes and compute).
# /p/project1 is a login-only mount; compute nodes cannot see it.

JEDI_PROJECT="${PROJECT_scifi:-/e/project1/scifi}"
JEDI_SCRATCH="${SCRATCH_scifi:-/e/scratch/scifi}"
JEDI_REPO_ROOT="${JEDI_REPO_ROOT:-${JEDI_PROJECT}/elsayed3/JEDi}"
# Backward-compatible alias used by older scripts.
JEDI_COMPUTE_ROOT="${JEDI_COMPUTE_ROOT:-${JEDI_REPO_ROOT}}"

translate_scifi_path() {
  local path="$1"
  path="${path//\/p\/project1\/scifi/${JEDI_PROJECT}}"
  path="${path//\/p\/scratch\/scifi/${JEDI_SCRATCH}}"
  echo "${path}"
}

resolve_jedi_repo_root() {
  if [[ -d "${JEDI_REPO_ROOT}" ]]; then
    echo "${JEDI_REPO_ROOT}"
    return 0
  fi

  if [[ -n "${REPO_ROOT:-}" ]]; then
    if [[ -d "${REPO_ROOT}" ]]; then
      echo "$(translate_scifi_path "${REPO_ROOT}")"
      return 0
    fi
    local translated
    translated="$(translate_scifi_path "${REPO_ROOT}")"
    if [[ -d "${translated}" ]]; then
      echo "${translated}"
      return 0
    fi
  fi

  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}" ]]; then
    echo "$(translate_scifi_path "${SLURM_SUBMIT_DIR}")"
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    local translated_submit
    translated_submit="$(translate_scifi_path "${SLURM_SUBMIT_DIR}")"
    if [[ -d "${translated_submit}" ]]; then
      echo "${translated_submit}"
      return 0
    fi
  fi

  echo "${JEDI_REPO_ROOT}"
}

require_jedi_repo() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo "ERROR: Repo not found at ${root}" >&2
    echo "Work under ${JEDI_REPO_ROOT} (not /p/project1)." >&2
    return 1
  fi
  if [[ ! -x "${root}/.venv/bin/python" ]]; then
    echo "ERROR: Missing venv at ${root}/.venv" >&2
    echo "Run: bash slurm_scripts/block_diffusion/setup_cuda_venv.sh" >&2
    return 1
  fi
  return 0
}

# Backward-compatible alias.
require_compute_visible_repo() {
  require_jedi_repo "$1"
}

activate_jupiter_modules() {
  set +u
  if ! command -v module >/dev/null 2>&1; then
    source /usr/share/lmod/lmod/init/bash 2>/dev/null \
      || source /etc/profile.d/lmod.sh 2>/dev/null \
      || true
  fi
  set -u
  module purge 2>/dev/null || true
  module load Stages/2026 GCC Python CUDA 2>/dev/null \
    || module load compiler/gnu/13 devel/cuda/12.4 2>/dev/null \
    || true
}
