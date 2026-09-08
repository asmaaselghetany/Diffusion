#!/usr/bin/env bash
# Create venv on login node (internet required). GH200 needs cu126 torch wheels.
set -euo pipefail

source /e/project1/scifi/elsayed3/env.sh
cd "${REPO_ROOT}"

module load Stages/2025 GCC Python CUDA 2>&1 | sed 's/^/[modules] /' || true

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e ".[dev]"

# aarch64 GH200: torch 2.6 + cu126 (2.7 has no cu126 wheel on aarch64)
pip uninstall -y torch torchvision 2>/dev/null || true
pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu126

echo "Done. Pre-download weights:"
echo "  HF_HOME=${HF_HOME} huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct"
