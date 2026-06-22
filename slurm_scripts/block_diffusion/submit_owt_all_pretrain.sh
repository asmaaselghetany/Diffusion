#!/usr/bin/env bash
# Submit all pretrain-init OWT baselines: MDLM (B) + AR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== MDLM pretrain (Baseline B) ==="
bash "${SCRIPT_DIR}/submit_owt_from_pretrain.sh"

echo ""
echo "=== AR pretrain (AR → block diffusion) ==="
bash "${SCRIPT_DIR}/submit_owt_from_ar_pretrain.sh"
