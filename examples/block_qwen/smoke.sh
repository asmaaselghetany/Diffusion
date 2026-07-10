#!/usr/bin/env bash
# Short GPU smoke: one training step per arm + verification scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH=src

if ! python -c "import torch; assert torch.cuda.is_available()"; then
  echo "CUDA required. Run on a GPU node."
  exit 1
fi

python scripts/smoke_train.py --algo block_masked --steps 1
python scripts/smoke_train.py --algo block_uniform --steps 1
python scripts/verify_qwen_load.py
python scripts/verify_block_forward.py
python scripts/verify_ar_block_init.py
python scripts/verify_block_sample.py
python scripts/smoke_loss_trend.py --steps 50

echo "block_qwen smoke PASS"
