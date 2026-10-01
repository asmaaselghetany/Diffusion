#!/usr/bin/env bash
#SBATCH --job-name=bqwen-revprobe
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Editable-token revision probe (uniform). Does not retrain.
#   CKPT=... sbatch --export=ALL,CKPT scripts/slurm/revision_probe.sbatch

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

CKPT="${CKPT:?Set CKPT}"
CKPT="$(readlink -f "${CKPT}")"
RUN_ROOT="$(dirname "$(dirname "${CKPT}")")"
OUT="${REVISION_OUT:-${RUN_ROOT}/eval/revision_probe}"
mkdir -p "${OUT}"

NUM_SAMPLES="${NUM_SAMPLES:-32}"
NUM_STEPS="${NUM_STEPS:-32}"
SAMPLE_MODE="${SAMPLE_MODE:-conversion_free}"
# Uniform: never bare baseline (full-seq Unif). generate_samples also coerces.
DECODE_PROFILE="${DECODE_PROFILE:-hierarchical_ss}"

echo "=== revision probe ==="
echo "  ckpt: ${CKPT}"
echo "  out:  ${OUT}"
echo "  decode_profile: ${DECODE_PROFILE}"

python -u -m discrete_diffusion.evaluations.generate_samples \
  "checkpoint_path=${CKPT}" \
  "samples_path=${OUT}/samples.pt" \
  "num_samples=${NUM_SAMPLES}" \
  batch_size=1 \
  "num_steps=${NUM_STEPS}" \
  "sample_mode=${SAMPLE_MODE}" \
  "decode_profile=${DECODE_PROFILE}" \
  max_new_tokens=512 \
  sampling.track_revisions=true \
  save_text=true \
  device=cuda

python -u -m discrete_diffusion.evaluations.generative_ppl \
  --config-name=gen_ppl_block_qwen \
  "samples_path=${OUT}/samples.pt" \
  "metrics_path=${OUT}/gen_ppl_metrics.json" \
  first_chunk_only=true

python tools/gen_ppl_hygiene.py pair "${OUT}/gen_ppl_metrics.json"
# Surface revision_rate from meta
python - "${OUT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
meta = json.loads((root / 'samples.meta.json').read_text())
rev = meta.get('revision_stats') or {}
(root / 'revision_stats.json').write_text(json.dumps(rev, indent=2) + '\n')
print('revision_stats:', rev)
PY
echo "Done ${OUT}"
