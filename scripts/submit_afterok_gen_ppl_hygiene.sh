#!/usr/bin/env bash
# Submit GenPPL+H hygiene that depends on a train job (CKPT may not exist yet).
#
# Usage:
#   ./scripts/submit_afterok_gen_ppl_hygiene.sh \
#     --dependency 2092151 \
#     --ckpt outputs/block_qwen/ar2block_uniform_2092151/checkpoints/last.ckpt
#
# Default MODE=dual (pair + H̄). Override MODE=all for primary-floor NFE sweeps.
set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh" 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/e/project1/scifi/elsayed3/Diffusion-new}"
cd "${REPO_ROOT}"

DEP=""
CKPT=""
JOB_NAME=""
MODE="${MODE:-dual}"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dependency|--dep) DEP="${2:?}"; shift 2 ;;
    --ckpt) CKPT="${2:?}"; shift 2 ;;
    --job-name) JOB_NAME="${2:?}"; shift 2 ;;
    --mode) MODE="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${DEP}" && -n "${CKPT}" ]] \
  || { echo "Required: --dependency --ckpt" >&2; exit 2; }

if [[ "${CKPT}" != /* ]]; then
  CKPT="${REPO_ROOT}/${CKPT}"
fi

RUN_ROOT="$(dirname "$(dirname "${CKPT}")")"
# Do not inherit ambient HYGIENE_OUT from other runs.
HYGIENE_OUT="${RUN_ROOT}/eval/unifusion_hygiene"
mkdir -p "${HYGIENE_OUT}"
JOB_NAME="${JOB_NAME:-atk-hyg-$(basename "${RUN_ROOT}" | tr '_' '-' | cut -c1-24)}"

echo "=== afterok GenPPL hygiene ==="
echo "  dep:  afterok:${DEP}"
echo "  ckpt: ${CKPT} (may be missing until train finishes)"
echo "  mode: ${MODE}"
echo "  out:  ${HYGIENE_OUT}"

if [[ "${DRY}" -eq 1 ]]; then
  echo "DRY: would sbatch --dependency=afterok:${DEP} gen_ppl_hygiene.sbatch"
  exit 0
fi

export CKPT MODE HYGIENE_OUT
export NUM_STEPS_LIST="${NUM_STEPS_LIST:-8 16 32 64}"
export RUN_MULTISEED="${RUN_MULTISEED:-0}"
export SAMPLE_MODE="${SAMPLE_MODE:-conversion_free}"
export DECODE_PROFILE="${DECODE_PROFILE:-hierarchical}"
export JOB_NAME

jid="$(sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --dependency="afterok:${DEP}" \
  --time="${TIME_LIMIT:-06:00:00}" \
  --export=ALL,CKPT,MODE,NUM_STEPS_LIST,NUM_SAMPLES,SAMPLE_MODE,DECODE_PROFILE,RUN_MULTISEED,SEEDS,MAX_NEW_TOKENS,HYGIENE_OUT,HOME \
  scripts/slurm/gen_ppl_hygiene.sbatch)"

echo "${jid}" > "${HYGIENE_OUT}/submit.jid"
cat > "${HYGIENE_OUT}/SUBMITTED.json" <<EOF
{
  "job_id": "${jid}",
  "dependency": "afterok:${DEP}",
  "ckpt": "${CKPT}",
  "mode": "${MODE}",
  "out_dir": "${HYGIENE_OUT}",
  "submitted_at": "$(date -Iseconds)"
}
EOF
echo "Submitted ${jid} (afterok:${DEP}) → ${HYGIENE_OUT}"
