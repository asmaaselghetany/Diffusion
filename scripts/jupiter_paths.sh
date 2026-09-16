#!/usr/bin/env bash
# Jupiter (JSC Booster) path helpers for Diffusion / block_qwen.
#
# Work only under /e/project1/scifi/... (visible on login nodes and compute).
# /p/project1 is login-only; compute nodes cannot see it.

SCIFI_PROJECT="${PROJECT_scifi:-/e/project1/scifi}"
SCIFI_SCRATCH="${SCRATCH_scifi:-/e/scratch/scifi}"
DIFFUSION_REPO_ROOT="${DIFFUSION_REPO_ROOT:-${SCIFI_PROJECT}/elsayed3/Diffusion-new}"
DIFFUSION_WORKSPACE="${DIFFUSION_WORKSPACE:-${SCIFI_PROJECT}/elsayed3}"

translate_scifi_path() {
  local path="$1"
  path="${path//\/p\/project1\/scifi/${SCIFI_PROJECT}}"
  path="${path//\/p\/scratch\/scifi/${SCIFI_SCRATCH}}"
  echo "${path}"
}

resolve_diffusion_repo_root() {
  if [[ -d "${DIFFUSION_REPO_ROOT}" ]]; then
    echo "${DIFFUSION_REPO_ROOT}"
    return 0
  fi
  if [[ -n "${REPO_ROOT:-}" ]]; then
    local translated
    translated="$(translate_scifi_path "${REPO_ROOT}")"
    if [[ -d "${translated}" ]]; then
      echo "${translated}"
      return 0
    fi
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    local submit_dir
    submit_dir="$(translate_scifi_path "${SLURM_SUBMIT_DIR}")"
    if [[ -d "${submit_dir}" ]]; then
      echo "${submit_dir}"
      return 0
    fi
  fi
  echo "${DIFFUSION_REPO_ROOT}"
}

require_diffusion_repo() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo "ERROR: Repo not found at ${root}" >&2
    echo "Work under ${DIFFUSION_REPO_ROOT} (not /p/project1)." >&2
    return 1
  fi
  if [[ ! -x "${root}/.venv/bin/python" ]]; then
    echo "ERROR: Missing venv at ${root}/.venv" >&2
    echo "Run on login node: bash scripts/setup_venv.sh" >&2
    return 1
  fi
  return 0
}

activate_jupiter_modules() {
  set +u
  if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null \
      || source /etc/profile.d/lmod.sh 2>/dev/null \
      || true
  fi
  set -u
  # Slurm often propagates the *parent* job's TMPDIR (e.g. train sets
  # /tmp/block_qwen_<train_jid>). Child evals then mktemp into a path that
  # does not exist on the new node → immediate FAIL (jobs 1774111/1774569).
  local _job="${SLURM_JOB_ID:-local}"
  local _want="/tmp/block_qwen_${_job}"
  if [[ -z "${TMPDIR:-}" || "${TMPDIR}" == /tmp/block_qwen_* ]]; then
    if [[ "${TMPDIR:-}" != "${_want}" || ! -d "${TMPDIR:-}" ]]; then
      export TMPDIR="${_want}"
    fi
  fi
  mkdir -p "${TMPDIR:-/tmp}" || export TMPDIR=/tmp
  local _modlog
  _modlog="$(mktemp "${TMPDIR}/modlog.XXXXXX")"
  module load Stages/2026 GCC Python CUDA >"${_modlog}" 2>&1 \
    || module load Stages/2025 GCC Python CUDA >"${_modlog}" 2>&1 \
    || true
  if [[ -s "${_modlog}" ]]; then
    sed 's/^/[modules] /' "${_modlog}"
  fi
  rm -f "${_modlog}"
}

# Booster allocates a full node (4 GH200 GPUs) per job.
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
