#!/usr/bin/env bash
# Submit masked + uniform Tiny Shakespeare smoke jobs on Booster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for mode in masked uniform; do
  echo "Submitting MODE=${mode} ..."
  MODE="${mode}" sbatch "${SCRIPT_DIR}/tiny_shakespeare.sh"
done
