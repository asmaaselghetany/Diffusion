#!/usr/bin/env bash
# Refuse submit/launch if forward_process/utils.py is not the tagged full blob.
# Source from _block_qwen_env.bash / slurm/_common.sh / submit_*.sh.
#
# Override: SKIP_UTILS_HASH_GUARD=1 (emergency only).
# Expected SHA is the baseline-ar2block-utils-full-2026-10-01 blob.

_EXPECTED_UTILS_SHA256="${EXPECTED_UTILS_SHA256:-1a3f6c494ed8bee8ea63fb66d7ff2b3e3f00cb3bf13555a3cb877749d1202f9e}"

assert_forward_process_utils_hash() {
  if [[ "${SKIP_UTILS_HASH_GUARD:-0}" == "1" ]]; then
    echo "[utils-hash] SKIP_UTILS_HASH_GUARD=1 — not checking" >&2
    return 0
  fi
  local root="${REPO_ROOT:-}"
  if [[ -z "${root}" ]]; then
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
  local path="${root}/src/discrete_diffusion/forward_process/utils.py"
  if [[ ! -f "${path}" ]]; then
    echo "FATAL: missing ${path} — refuse submit (expected full Unif utils)" >&2
    return 99
  fi
  local got
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "${path}" | awk '{print $1}')"
  else
    got="$(openssl dgst -sha256 "${path}" | awk '{print $NF}')"
  fi
  local nlines
  nlines="$(wc -l < "${path}" | tr -d ' ')"
  if [[ "${got}" != "${_EXPECTED_UTILS_SHA256}" ]]; then
    echo "FATAL: forward_process/utils.py hash mismatch" >&2
    echo "  path:     ${path}" >&2
    echo "  got:      ${got} (${nlines} lines)" >&2
    echo "  expected: ${_EXPECTED_UTILS_SHA256} (tag baseline-ar2block-utils-full-2026-10-01)" >&2
    echo "  hint:     clean clones still have the 45-line stub; checkout the tag or restore the full file." >&2
    return 99
  fi
  if [[ "${nlines}" -lt 100 ]]; then
    echo "FATAL: utils.py looks stubby (${nlines} lines) despite hash match?" >&2
    return 99
  fi
  echo "[utils-hash] ok sha256=${got} lines=${nlines} root=${root}"
  return 0
}
