#!/usr/bin/env bash
# Post-train sample + gen-PPL eval (UNI-D² generate_samples + generative_ppl).
#
# Defaults (adopted policy):
#   sample_mode=auto          — conversion_free (ar2block/Nemotron) vs native_free (OWT/block)
#   decode_profile=baseline   — coerces → hierarchical (thr=0.9 greedy remask floor)
#   max_new_tokens=512        — do not fill full 2048 by default
#
# Refuse reuse of samples.pt unless samples.meta.json matches mode+profile+ckpt.
#   FORCE_REGEN=1  — always regenerate
#
# Usage:
#   bash examples/block_qwen/eval.sh <checkpoint.ckpt> [eval_dir]
#   SAMPLE_MODE=bare_bos DECODE_PROFILE=keep bash examples/block_qwen/eval.sh <ckpt>
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
SAMPLE_MODE="${SAMPLE_MODE:-auto}"
DECODE_PROFILE="${DECODE_PROFILE:-baseline}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
FORCE_REGEN="${FORCE_REGEN:-0}"

echo "checkpoint: ${CKPT}"
echo "eval_dir:   ${EVAL_DIR}"
echo "sample:     mode=${SAMPLE_MODE} decode_profile=${DECODE_PROFILE} max_new=${MAX_NEW_TOKENS}"

REUSE=0
if [[ "${FORCE_REGEN}" != "1" ]]; then
  if python -c "
from discrete_diffusion.evaluations.decode_profiles import should_reuse_samples
import sys
ok = should_reuse_samples(
    '${SAMPLES_PT}',
    checkpoint_path='${CKPT}',
    sample_mode='${SAMPLE_MODE}',
    decode_profile='${DECODE_PROFILE}',
    force_regen=False,
)
sys.exit(0 if ok else 1)
"; then
    REUSE=1
  fi
fi

if [[ "${REUSE}" -eq 1 ]]; then
  echo "Reusing samples (meta gate matched): ${SAMPLES_PT}"
else
  if [[ -f "${SAMPLES_PT}" ]]; then
    echo "Regenerating samples (stale/missing meta or FORCE_REGEN=1)"
  fi
  python -u -m discrete_diffusion.evaluations.generate_samples \
    "checkpoint_path=${CKPT}" \
    "samples_path=${SAMPLES_PT}" \
    "num_samples=${NUM_SAMPLES}" \
    "batch_size=${BATCH_SIZE}" \
    "num_steps=${NUM_STEPS}" \
    "sample_mode=${SAMPLE_MODE}" \
    "decode_profile=${DECODE_PROFILE}" \
    "max_new_tokens=${MAX_NEW_TOKENS}" \
    save_text=true \
    device=cuda
fi

python -u -m discrete_diffusion.evaluations.generative_ppl \
  --config-name=gen_ppl_block_qwen \
  "samples_path=${SAMPLES_PT}" \
  "metrics_path=${METRICS_JSON}" \
  pretrained_model=gpt2-large \
  retokenize=true \
  first_chunk_only="${FIRST_CHUNK_ONLY:-true}"

# Unifusion hygiene: GenPPL must travel with unigram entropy.
python tools/gen_ppl_hygiene.py pair "${METRICS_JSON}" \
  || echo "WARNING: gen_ppl_pair sidecar failed (missing entropy?)" >&2

echo "Done. See ${EVAL_DIR} (samples.pt, samples.txt, samples.meta.json, gen_ppl_metrics.json, gen_ppl_pair.json)"
