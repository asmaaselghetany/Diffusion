#!/usr/bin/env bash
# C4: NFE sweep — now routes through Unifusion GenPPL hygiene (PPL+H pair).
#
# Legacy name kept for submit_paper_cell.sh C4.
# Prefer: CKPT=... ./scripts/submit_gen_ppl_hygiene.sh
#
# Usage:
#   CKPT=outputs/.../last.ckpt ./scripts/submit_nfe_sweep.sh
#   CKPT=... NUM_STEPS_LIST="8 16 32 64" ./scripts/submit_nfe_sweep.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT:-${1:-}}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Usage: CKPT=/path/to.ckpt $0" >&2
  exit 1
fi
export CKPT="$(readlink -f "${CKPT}")"
export MODE="${MODE:-nfe}"
export NUM_STEPS_LIST="${NUM_STEPS_LIST:-8 16 32 64}"
export SAMPLE_MODE="${SAMPLE_MODE:-conversion_free}"
export DECODE_PROFILE="${DECODE_PROFILE:-hierarchical_ss}"
# Default: NFE + collapse panel via MODE=all would regenerate heavily;
# C4 is NFE-focused. Panel runs if default eval artifacts exist after sweep.
export MODE="${MODE:-nfe}"

echo "=== C4 NFE → GenPPL hygiene submit ==="
echo "  decode_profile=${DECODE_PROFILE} (uniform-hostile baseline remapped in hygiene)"
bash "${REPO_ROOT}/scripts/submit_gen_ppl_hygiene.sh"
