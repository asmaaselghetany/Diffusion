#!/usr/bin/env bash
# Pre-download Qwen weights + Nemotron SFT on the login node (internet required).
# Compute nodes run with HF_HUB_OFFLINE=1.
set -euo pipefail

source /e/project1/scifi/elsayed3/env.sh
cd "${REPO_ROOT}"

module load Stages/2025 GCC Python CUDA 2>&1 | sed 's/^/[modules] /' || true
source .venv/bin/activate

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Remove stale partial caches from failed compute attempts.
find "${DATA_CACHE}" -name '*.incomplete' -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Qwen2.5-1.5B-Instruct → ${HF_HOME} ==="
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct

echo "=== gpt2-large (gen-PPL judge) → ${HF_HOME} ==="
huggingface-cli download openai-community/gpt2-large || huggingface-cli download gpt2-large

echo "=== Nemotron SFT + wrapped .dat caches → ${DATA_CACHE} ==="
python -u tools/prefetch_nemotron.py

echo "=== Done. Cache sizes ==="
du -sh "${HF_HOME}" "${DATA_CACHE}" 2>/dev/null || true
