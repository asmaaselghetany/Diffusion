#!/usr/bin/env bash
# Submit Pipeline 1: AR→block (pretrained Qwen → block SFT, hooks off).
#
# Usage:
#   ./scripts/submit_ar2block.sh both
#   ./scripts/submit_ar2block.sh masked
#   ./scripts/submit_ar2block.sh uniform

set -euo pipefail

WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

submit() {
  local arm="$1"
  local script="scripts/slurm/ar2block_${arm}.sbatch"
  if [[ ! -f "${script}" ]]; then
    echo "Missing ${script}" >&2
    exit 1
  fi
  echo "Submitting ${script} ..."
  sbatch "${script}"
}

TARGET="${1:-both}"

case "${TARGET}" in
  masked|uniform)
    submit "${TARGET}"
    ;;
  all|both)
    submit masked
    submit uniform
    ;;
  *)
    echo "Usage: $0 {masked|uniform|both}" >&2
    exit 1
    ;;
esac
