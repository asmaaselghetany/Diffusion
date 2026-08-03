#!/usr/bin/env bash
# Resolve block_qwen run directories and checkpoint paths.
#
# Lines:
#   ar2block  — Pipeline 1 AR→block: ar2block_{masked,uniform}_<jobid>
#   block     — Pipeline 2 pure block: block_{masked,uniform}_<jobid>
#   blockgen  — legacy alias for block: blockgen_{masked,uniform}_<jobid>
# Legacy (Pipeline 1):
#   masked_<jobid> / uniform_<jobid>
#
# Usage:
#   source scripts/_resolve_block_qwen_run.bash
#   resolve_block_qwen_ckpt ar2block masked 126234
#   resolve_block_qwen_ckpt block uniform 126999
#   resolve_block_qwen_ckpt masked 126234   # legacy

set -euo pipefail

_resolve_block_qwen_base() {
  if [[ -n "${UNI_D2_ROOT:-}" ]]; then
    echo "${UNI_D2_ROOT}/outputs/block_qwen"
    return
  fi
  local workspace="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
  echo "${workspace}/uni-d2/outputs/block_qwen"
}

resolve_block_qwen_run_dir() {
  local line arm job_id
  local base
  base="$(_resolve_block_qwen_base)"

  if [[ "${1:-}" =~ ^(ar2block|block|blockgen)$ ]]; then
    line="${1}"
    arm="${2:?arm required (masked|uniform)}"
    job_id="${3:?job_id required}"
  elif [[ "${1:-}" =~ ^(neutral|auto)$ ]]; then
    # Legacy callers: prefer legacy dir, then ar2block.
    arm="${2:?arm required (masked|uniform)}"
    job_id="${3:?job_id required}"
    if [[ -d "${base}/${arm}_${job_id}" ]]; then
      echo "${base}/${arm}_${job_id}"
      return 0
    fi
    echo "${base}/ar2block_${arm}_${job_id}"
    return 0
  else
    # Two-arg form: arm job_id → legacy or ar2block
    arm="${1:?arm required (masked|uniform)}"
    job_id="${2:?job_id required}"
    if [[ -d "${base}/${arm}_${job_id}" ]]; then
      echo "${base}/${arm}_${job_id}"
      return 0
    fi
    echo "${base}/ar2block_${arm}_${job_id}"
    return 0
  fi

  if [[ "${arm}" != "masked" && "${arm}" != "uniform" ]]; then
    echo "Unknown arm: ${arm} (expected masked|uniform)" >&2
    return 1
  fi

  echo "${base}/${line}_${arm}_${job_id}"
}

resolve_block_qwen_ckpt() {
  local run_dir
  run_dir="$(resolve_block_qwen_run_dir "$@")"
  echo "${run_dir}/checkpoints/last.ckpt"
}

# Prints: LINE ARM JOB_ID
# LINE is ar2block|block|blockgen; legacy masked_/uniform_ → ar2block
parse_block_qwen_run_basename() {
  local name="${1:?}"
  if [[ "${name}" =~ ^(ar2block|block|blockgen)_(masked|uniform)_([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}"
    return 0
  fi
  if [[ "${name}" =~ ^(masked|uniform)_([0-9]+)$ ]]; then
    echo "ar2block ${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    return 0
  fi
  echo "Unrecognized run directory basename: ${name}" >&2
  echo "Expected: ar2block_{masked,uniform}_<id>, block_{masked,uniform}_<id>," >&2
  echo "          blockgen_{masked,uniform}_<id> (legacy), or masked_<id> / uniform_<id>" >&2
  return 1
}
