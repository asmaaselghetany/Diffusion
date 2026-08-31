#!/usr/bin/env bash
# Submit Slurm job(s) for thesis longitudinal eval (PAPER_EXPERIMENTS §7).
#
# Usage:
#   ./scripts/submit_longitudinal_eval.sh outputs/block_qwen/ar2block_masked_141728
#   ./scripts/submit_longitudinal_eval.sh outputs/block_qwen/ar_sft_143599
#   STEPS="500 1000" ./scripts/submit_longitudinal_eval.sh <run_root>
#
# Optional env:
#   STEPS       — space-separated milestones (default: 0 500 1000 2000 4000 6000)
#   SKIP_INIT   — set 1 to skip step_0 retention bundle
#   DRY_RUN     — print plan only

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

RUN_ROOT="${1:?Usage: submit_longitudinal_eval.sh <run_root>}"
RUN_ROOT="$(readlink -f "${RUN_ROOT}")"
STEPS="${STEPS:-0 500 1000 2000 4000 6000}"

echo "=== longitudinal eval plan ==="
echo "  run_root: ${RUN_ROOT}"
echo "  steps:    ${STEPS}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — would submit:"
  echo "  python tools/run_milestone_eval.py --run-root ${RUN_ROOT} --steps ${STEPS}"
  exit 0
fi

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

export RUN_ROOT STEPS SKIP_INIT="${SKIP_INIT:-0}"
srun python -u tools/run_milestone_eval.py \
  --run-root "${RUN_ROOT}" \
  --steps ${STEPS} \
  ${SKIP_INIT:+--skip-init}

echo "Longitudinal bundles under ${RUN_ROOT}/eval/step_*/"
