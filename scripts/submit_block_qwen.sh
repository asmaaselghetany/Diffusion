#!/usr/bin/env bash
# Submit block_qwen training — forwards to the two pipeline submitters.
#
# Usage:
#   ./scripts/submit_block_qwen.sh ar2block both   # AR→block
#   ./scripts/submit_block_qwen.sh block both      # scratch block
#   ./scripts/submit_block_qwen.sh                 # prints help

set -euo pipefail

WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

LINE="${1:-}"
TARGET="${2:-both}"

case "${LINE}" in
  ar2block|p1|fastdllm)
    exec ./scripts/submit_ar2block.sh "${TARGET}"
    ;;
  block|blockgen|p2)
    # blockgen = legacy alias for scratch "block" arms
    exec ./scripts/submit_block.sh "${TARGET}"
    ;;
  *)
    cat <<'EOF' >&2
Usage: ./scripts/submit_block_qwen.sh <pipeline> [masked|uniform|both]

Pipelines:
  ar2block   AR→block (pretrained Qwen → block SFT)
  block      Scratch block diffusion (same arch, no AR weights)

Examples:
  ./scripts/submit_ar2block.sh both
  ./scripts/submit_block.sh both
  ./scripts/submit_block_qwen.sh ar2block both
  ./scripts/submit_block_qwen.sh block both
EOF
    exit 1
    ;;
esac
