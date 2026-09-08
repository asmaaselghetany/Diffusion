#!/usr/bin/env bash
# BlockGen OWT uniform 1+16 — one-to-one recipe on our Qwen components.
#
# Official ref: third_party/blockgen/scripts/train/owt/blockgen_uniform_1_16.sh
#   data=openwebtext-split (jdeschena/openwebtext)
#   algo=blockgen-uniform, elbo + CE@1, pure_noise=[1]
#   block_weights 0.05…0.95 (sizes 1 & 16), u-stratified
#   GBS 512, length 1024, lr 3e-4, 1M steps, scratch Block-DiT
#
# Our wiring:
#   line=block (scratch Qwen), algo=block_uniform
#   data=openwebtext-blockgen (same HF + splits; Qwen tokenizer)
#   levers = bg_weights_1_16 + u_stratified + pure_noise_1 + ce_at_1 + arpc_blockgen
#
# Honest deltas (cannot be bit-identical): backbone Qwen-1.5B vs 170M Block-DiT;
# tokenizer Qwen vs GPT-2; wall-clock may need AUTO_RESUME toward 1M steps.
#
# Usage:
#   ./scripts/submit_blockgen_owt.sh              # paper 8×4, submit
#   ./scripts/submit_blockgen_owt.sh --dry-run
#   PREFETCH_ONLY=1 ./scripts/submit_blockgen_owt.sh

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

DRY_ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_ARGS+=(--dry-run) ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
  esac
done

# Always use BlockGen OWT cache (do not inherit Nemotron DATA_CACHE from env.sh).
export DATA_CACHE="${BLOCKGEN_OWT_CACHE:-${DISCRETE_DIFFUSION_SCRATCH_DIR}/openwebtext-blockgen-qwen}"
mkdir -p "${DATA_CACHE}"

if [[ "${PREFETCH_ONLY:-0}" == "1" ]] || [[ "${SKIP_PREFETCH:-0}" != "1" ]]; then
  echo "=== Prefetch BlockGen OWT + Qwen wrap caches → ${DATA_CACHE} ==="
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
  fi
  export HF_HUB_OFFLINE=0
  export HF_DATASETS_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  # Dry-run skips the multi-GB download unless FORCE_PREFETCH=1.
  if [[ " ${*:-} " == *" --dry-run "* ]] && [[ "${FORCE_PREFETCH:-0}" != "1" ]]; then
    echo "(dry-run) skipping prefetch; set FORCE_PREFETCH=1 to download anyway"
  else
    BLOCKGEN_OWT_SEQ_LEN="${BLOCKGEN_OWT_SEQ_LEN:-1024}" \
    BLOCKGEN_OWT_BLOCK="${BLOCKGEN_OWT_BLOCK:-16}" \
    DATA_CACHE="${DATA_CACHE}" \
      python -u tools/prefetch_openwebtext_blockgen.py
  fi
  if [[ "${PREFETCH_ONLY:-0}" == "1" ]]; then
    echo "PREFETCH_ONLY=1 → exiting before submit"
    exit 0
  fi
fi

# Recipe-matched train knobs (override paper 2048/32/256/2e-5 defaults).
export BLOCK="${BLOCK:-16}"
export SEQ_LEN="${SEQ_LEN:-1024}"
export GBS="${GBS:-512}"
export MAX_STEPS="${MAX_STEPS:-1000000}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export NUM_NODES="${NUM_NODES:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export WANDB_PROJECT="${WANDB_PROJECT:-block_qwen}"

# Non-lever Hydra: data + BlockGen optim/lr + checkpoint cadence (opt steps).
# Keep block_qwen val_check_interval (500 * accumulate) — do not clobber with
# a raw batch count.
_extra_base="data=openwebtext-blockgen data.tokenizer_name_or_path=Qwen/Qwen2.5-1.5B-Instruct optim.lr=3e-4 callbacks.checkpoint_every_n_steps.every_n_train_steps=5000"
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  export EXTRA_OVERRIDES="${EXTRA_OVERRIDES} ${_extra_base}"
else
  export EXTRA_OVERRIDES="${_extra_base}"
fi

exec ./scripts/submit_lever.sh \
  --preset blockgen_owt_uniform \
  --arm uniform \
  --line block \
  --paper \
  "${DRY_ARGS[@]}"
