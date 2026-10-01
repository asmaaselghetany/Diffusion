#!/usr/bin/env bash
#SBATCH --job-name=bqwen-arpc
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Decode-only BlockGen ARPC sample regen on an existing checkpoint.
# Writes under EVAL_RUN_DIR (does not overwrite the baseline eval/).
#
# Pins match lever ``arpc_blockgen`` (BlockGen TinyGSM absorb/uniform scripts):
#   use_arpc=true arpc_mode=blockgen corruption=ar_metric ar_metric=nll
#   prefix_fill=false
#
# Prerequisite: ckpt trained with algo.block_size_mixture / block_weights
# including size 1 (BlockGen AR verify L'=1). ``submit_family_eval`` skips ARPC
# when EVAL_HAS_ARPC_SIZE1=0 unless FORCE_ARPC=1. Decode-only on fixed-size
# ckpts warns (no hard fail) and is expected to be weak.
#
# Usage:
#   sbatch scripts/slurm/arpc_decode_eval.sbatch <checkpoint> <eval_run_dir> [num_samples]
# Optional env: ARPC_MODE, ARPC_CORRUPTION, ARPC_AR_METRIC, ARPC_PREFIX_FILL

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

CKPT="${1:?Usage: arpc_decode_eval.sbatch <checkpoint> <eval_run_dir> [num_samples]}"
OUT="${2:?Usage: arpc_decode_eval.sbatch <checkpoint> <eval_run_dir> [num_samples]}"
NUM_SAMPLES="${3:-8}"
mkdir -p "${OUT}" slurm_logs

CKPT_ABS="$(readlink -f "${CKPT}")"
SAMPLES_PT="${OUT}/samples.pt"
SAMPLES_TXT="${OUT}/samples.txt"

# Allow env override for transfer/simplified probes; default = BlockGen paper pin.
ARPC_MODE="${ARPC_MODE:-blockgen}"
ARPC_CORRUPTION="${ARPC_CORRUPTION:-ar_metric}"
ARPC_AR_METRIC="${ARPC_AR_METRIC:-nll}"
ARPC_PREFIX_FILL="${ARPC_PREFIX_FILL:-false}"

echo "=== ARPC decode-only ==="
echo "  ckpt: ${CKPT_ABS}"
echo "  out:  ${OUT}"
echo "  num_samples: ${NUM_SAMPLES}"
echo "  pins: use_arpc=true arpc_mode=${ARPC_MODE} corruption=${ARPC_CORRUPTION} ar_metric=${ARPC_AR_METRIC} prefix_fill=${ARPC_PREFIX_FILL}"

# Older ckpt hydra dumps may omit ARPC keys (struct) — use + to add/override.
# Force schedule decode (no Hub confidence unmask) so masked ARPC can run.
srun python -u -m discrete_diffusion.evaluations.generate_samples \
  "checkpoint_path=${CKPT_ABS}" \
  "samples_path=${SAMPLES_PT}" \
  "num_samples=${NUM_SAMPLES}" \
  batch_size=1 \
  num_steps=32 \
  save_text=true \
  device=cuda \
  sample_mode=auto \
  decode_profile=keep \
  "+sampling.use_arpc=true" \
  "+sampling.arpc_mode=${ARPC_MODE}" \
  "+sampling.arpc_corruption_mode=${ARPC_CORRUPTION}" \
  "+sampling.arpc_ar_metric=${ARPC_AR_METRIC}" \
  "+sampling.arpc_use_prefix_fill=${ARPC_PREFIX_FILL}" \
  "+sampling.unmask_threshold=null"

# Optional gen-PPL if gpt2-large is cached.
if python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('gpt2-large', local_files_only=True)" 2>/dev/null; then
  srun python -u -m discrete_diffusion.evaluations.generative_ppl \
    --config-name=gen_ppl_block_qwen \
    "samples_path=${SAMPLES_PT}" \
    "metrics_path=${OUT}/gen_ppl_metrics.json" \
    pretrained_model=gpt2-large \
    retokenize=true \
    first_chunk_only=true || echo "WARNING: gen-PPL failed" >&2
else
  echo "WARNING: gpt2-large not cached; skipping gen-PPL" >&2
fi

echo "Done. Samples: ${SAMPLES_TXT}"
