#!/usr/bin/env bash
# Submit editable-token revision probe (1 GPU, no train).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
CKPT="${CKPT:-${1:-}}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Usage: CKPT=/path/to.ckpt $0" >&2
  exit 1
fi
export CKPT="$(readlink -f "${CKPT}")"
sbatch --export=ALL,CKPT,REVISION_OUT,NUM_SAMPLES,NUM_STEPS,SAMPLE_MODE \
  scripts/slurm/revision_probe.sbatch
