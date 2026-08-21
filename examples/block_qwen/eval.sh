#!/usr/bin/env bash
# Post-train eval using UNI-D² components only (generate_samples + generative_ppl).
# Usage:
#   bash examples/block_qwen/eval.sh outputs/block_qwen/ar2block_masked/checkpoints/last.ckpt
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="src:${PYTHONPATH:-}"

CKPT="${1:?Usage: $0 <checkpoint.ckpt> [eval_dir]}"
EVAL_DIR="${2:-$(dirname "$(dirname "${CKPT}")")/eval}"
mkdir -p "${EVAL_DIR}"

SAMPLES_PT="${EVAL_DIR}/samples.pt"
METRICS_JSON="${EVAL_DIR}/gen_ppl_metrics.json"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_STEPS="${NUM_STEPS:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"

echo "checkpoint: ${CKPT}"
echo "eval_dir:   ${EVAL_DIR}"

if [[ ! -f "${SAMPLES_PT}" ]]; then
  python -u -m discrete_diffusion.evaluations.generate_samples \
    "checkpoint_path=${CKPT}" \
    "samples_path=${SAMPLES_PT}" \
    "num_samples=${NUM_SAMPLES}" \
    "batch_size=${BATCH_SIZE}" \
    "num_steps=${NUM_STEPS}" \
    save_text=true \
    device=cuda
else
  echo "Reusing existing samples: ${SAMPLES_PT}"
fi

python -u -m discrete_diffusion.evaluations.generative_ppl \
  --config-name=gen_ppl_block_qwen \
  "samples_path=${SAMPLES_PT}" \
  "metrics_path=${METRICS_JSON}" \
  pretrained_model=gpt2-large \
  retokenize=true \
  first_chunk_only=true

echo "Wrote ${SAMPLES_PT} and ${METRICS_JSON}"
echo "Optional fork extras (DepBench / block ELBO):"
echo "  python tools/run_block_qwen_eval.py --checkpoint ${CKPT} --skip-samples --skip-gen-ppl"
