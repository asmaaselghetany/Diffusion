#!/usr/bin/env bash
# Install CUDA-enabled PyTorch into the project venv (login node only — has internet).
#
# Jupiter Booster compute nodes are offline. The venv must be built on the login
# node with a CUDA wheel before SLURM jobs can use GPUs.
#
# On aarch64 (GH200), PyTorch publishes cu126 wheels for cp39 up to torch 2.6.
#
# Usage:
#   bash slurm_scripts/block_diffusion/setup_cuda_venv.sh
#   bash slurm_scripts/block_diffusion/setup_compute_workspace.sh   # sync to /e
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  echo "ERROR: Create .venv first (python -m venv .venv && pip install -r requirements.txt)" >&2
  exit 1
fi

set +u
source "${REPO_ROOT}/.venv/bin/activate"
set -u

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
ARCH="$(python -c 'import platform; print(platform.machine())')"
echo "Python ${PY_VER} on ${ARCH}"

echo "Installing torch with CUDA (cu126 index)..."
pip uninstall -y torch 2>/dev/null || true
if ! pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu126; then
  echo "ERROR: Failed to install CUDA torch. Check Python version / architecture." >&2
  exit 1
fi
pip install "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu126

# Triton for flex attention (not on PyPI for aarch64 — use PyTorch's cu126 index).
echo "Installing triton (required for model.attn_backend=flex)..."
pip install triton --index-url https://download.pytorch.org/whl/cu126

python -c "
from discrete_diffusion.compat.triton_shim import ensure_triton_attrs_descriptor
ensure_triton_attrs_descriptor()
import torch
import triton
from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
from discrete_diffusion.models.block_dit import get_flex_runtime
print('torch', torch.__version__, 'cuda built', torch.version.cuda, 'available', torch.cuda.is_available())
print('triton', triton.__version__, 'flex_attention OK')
print('flex_runtime', get_flex_runtime())
if '+cpu' in torch.__version__:
    raise SystemExit('ERROR: still CPU-only torch')
"

echo ""
echo "Done. Sync to compute path:"
echo "  bash slurm_scripts/block_diffusion/setup_compute_workspace.sh"
