#!/usr/bin/env bash
# Submit IFEval-only lm_eval with --log_samples for editable-token Phase 1.
# Does not touch C0/U0 canonical SUMMARY trees (separate OUT_DIR).
#
# Usage:
#   ./scripts/submit_ifeval_samples.sh C0
#   ./scripts/submit_ifeval_samples.sh U0
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

CELL="${1:?Usage: $0 C0|U0}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

case "${CELL}" in
  C0)
    CKPT="${CKPT:-/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt}"
    JOB_TAG=1762534
    DECODE_PROFILE="${DECODE_PROFILE:-hubmatch}"
    UNMASK_THRESHOLD="${UNMASK_THRESHOLD:-1}"
    FORCE_GREEDY="${FORCE_GREEDY:-1}"
    RUN_ROOT="$(dirname "$(dirname "${CKPT}")")"
    OUT_DIR="${OUT_DIR:-${RUN_ROOT}/lm_eval_ifeval_samples_hubmatch}"
    ;;
  U0)
    # Claim U0 (not contaminated 1849335). Bake twin floor = UC.
    CKPT="${CKPT:-/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt}"
    JOB_TAG=1955203
    DECODE_PROFILE="${DECODE_PROFILE:-uniform_commit}"
    UNMASK_THRESHOLD="${UNMASK_THRESHOLD:-}"
    FORCE_GREEDY="${FORCE_GREEDY:-}"
    RUN_ROOT="$(dirname "$(dirname "${CKPT}")")"
    OUT_DIR="${OUT_DIR:-${RUN_ROOT}/lm_eval_ifeval_samples_uniform_commit}"
    ;;
  *)
    echo "Unknown cell ${CELL} (C0|U0)" >&2
    exit 2
    ;;
esac

[[ -f "${CKPT}" ]] || { echo "Missing ckpt ${CKPT}" >&2; exit 1; }

export CKPT OUT_DIR
export SUITE=custom
export TASKS=ifeval
export DECODE_PROFILE
export FORCE_GREEDY
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export NUM_STEPS="${NUM_STEPS:-32}"
export SKIP_THROUGHPUT=1
export LOG_SAMPLES=1
export NUM_NODES GPUS_PER_NODE
if [[ -n "${UNMASK_THRESHOLD}" ]]; then
  export UNMASK_THRESHOLD
else
  unset UNMASK_THRESHOLD || true
fi

echo "Submitting IFEval+log_samples for ${CELL} (job_tag=${JOB_TAG})"
echo "  ckpt=${CKPT}"
echo "  out=${OUT_DIR}"
echo "  profile=${DECODE_PROFILE} thr=${UNMASK_THRESHOLD:-none} greedy=${FORCE_GREEDY}"
echo "  nodes=${NUM_NODES}x${GPUS_PER_NODE}"

env -u TMPDIR -u TEMP -u TMP \
  sbatch --parsable --nodes="${NUM_NODES}" \
  --job-name="ifeval_${CELL}" \
  --export=ALL \
  scripts/slurm/lm_eval.sbatch
