#!/usr/bin/env bash
# Submit Unifusion-style GenPPL hygiene jobs (C4 / dual / NFE / panel).
#
#   CKPT=... ./scripts/submit_gen_ppl_hygiene.sh
#   CKPT=... MODE=dual ./scripts/submit_gen_ppl_hygiene.sh
#   CKPT=... RUN_MULTISEED=1 ./scripts/submit_gen_ppl_hygiene.sh
#
# Optional base-LM slice (separate lm_eval job):
#   CKPT=... ./scripts/submit_gen_ppl_hygiene.sh --with-winogrande

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

WITH_WINO=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--with-winogrande" || "$a" == "--with-hellaswag" ]]; then
    WITH_WINO=1
    EXTRA_TASK="${a#--with-}"
  else
    ARGS+=("$a")
  fi
done

CKPT="${CKPT:-${ARGS[0]:-}}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Usage: CKPT=/path/to.ckpt $0 [--with-winogrande|--with-hellaswag]" >&2
  exit 1
fi
CKPT="$(readlink -f "${CKPT}")"
MODE="${MODE:-all}"
export CKPT MODE
export NUM_STEPS_LIST="${NUM_STEPS_LIST:-8 16 32 64}"
export RUN_MULTISEED="${RUN_MULTISEED:-0}"
export SAMPLE_MODE="${SAMPLE_MODE:-conversion_free}"

echo "=== submit GenPPL hygiene ==="
echo "  ckpt: ${CKPT}"
echo "  mode: ${MODE}"
echo "  steps:${NUM_STEPS_LIST}"
echo "  multiseed: ${RUN_MULTISEED}"

sbatch --export=ALL,CKPT,MODE,NUM_STEPS_LIST,NUM_SAMPLES,SAMPLE_MODE,DECODE_PROFILE,RUN_MULTISEED,SEEDS,MAX_NEW_TOKENS,HYGIENE_OUT,HOME \
  --job-name="bqwen-hyg-$(basename "$(dirname "$(dirname "${CKPT}")")" | tr '_' '-' | cut -c1-20)" \
  scripts/slurm/gen_ppl_hygiene.sbatch

if [[ "${WITH_WINO}" -eq 1 ]]; then
  TASK="${EXTRA_TASK:-winogrande}"
  echo "=== optional base-LM slice: ${TASK} ==="
  export TASKS="${TASK}"
  export SUITE="${SUITE:-base_lm_slice}"
  export NUM_NODES="${NUM_NODES:-1}"
  export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
  sbatch --export=ALL,CKPT,TASKS,SUITE,NUM_NODES,GPUS_PER_NODE,NUM_STEPS,MAX_NEW_TOKENS,SKIP_THROUGHPUT,OUT_DIR,DECODE_PROFILE \
    --nodes=1 --ntasks-per-node=4 --gres=gpu:4 --time=04:00:00 \
    --job-name="bqwen-${TASK}" \
    scripts/slurm/lm_eval.sbatch
fi
