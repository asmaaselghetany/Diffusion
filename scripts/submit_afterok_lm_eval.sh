#!/usr/bin/env bash
# Submit lm_eval that depends on a train job finishing, with a CKPT path that
# does not exist yet (resolved at batch start after afterok).
#
# submit_family_eval.sh refuses missing ckpts at submit time — that forced
# either waiting until train ends (wasted queue latency) or pointing at a
# wrong existing ckpt. This helper is the only safe afterok path.
#
# Usage:
#   ./scripts/submit_afterok_lm_eval.sh \
#     --dependency 2011999 \
#     --ckpt outputs/block_qwen/ar2block_masked_2011999/checkpoints/last.ckpt \
#     --out  outputs/block_qwen/ar2block_masked_2011999/lm_eval_... \
#     --profile hierarchical_quiet --steps 8 --arm masked
set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh" 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/e/project1/scifi/elsayed3/Diffusion-new}"
cd "${REPO_ROOT}"

DEP=""
CKPT=""
OUT=""
PROFILE=""
STEPS=32
ARM=""   # masked|uniform — picks stack/suite defaults
GREEDY_PIN=""
THR=""
JOB_NAME=""
# Full paper suite by default (paper_acc / paper_gen → includes MMLU + GSM + IFE).
# Do NOT inherit shell TASKS/SUITE — only --tasks / --suite (avoids gsm8k-only footgun).
TASKS_OVERRIDE=""
SUITE_OVERRIDE=""
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_NODES="${NUM_NODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dependency|--dep) DEP="${2:?}"; shift 2 ;;
    --ckpt) CKPT="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --steps) STEPS="${2:?}"; shift 2 ;;
    --arm) ARM="${2:?}"; shift 2 ;;
    --greedy-pin) GREEDY_PIN="${2:?}"; shift 2 ;;
    --thr) THR="${2:?}"; shift 2 ;;
    --job-name) JOB_NAME="${2:?}"; shift 2 ;;
    --suite) SUITE_OVERRIDE="${2:?}"; shift 2 ;;
    --tasks) TASKS_OVERRIDE="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${DEP}" && -n "${CKPT}" && -n "${OUT}" && -n "${PROFILE}" && -n "${ARM}" ]] \
  || { echo "Required: --dependency --ckpt --out --profile --arm" >&2; exit 2; }

case "${ARM}" in
  masked)
    STACK=conversion_lm_eval
    SUITE="${SUITE_OVERRIDE:-paper_acc}"
    EVAL_FORWARD=masked
    ;;
  uniform|hybrid)
    STACK=blockgen_arpc
    SUITE="${SUITE_OVERRIDE:-paper_gen}"
    EVAL_FORWARD="${ARM}"
    ;;
  *) echo "arm must be masked|uniform|hybrid" >&2; exit 2 ;;
esac

JOB_NAME="${JOB_NAME:-afterok-${ARM}-${PROFILE}-s${STEPS}}"
mkdir -p "${OUT}"

# Absolute ckpt path (may not exist yet).
if [[ "${CKPT}" != /* ]]; then
  CKPT="${REPO_ROOT}/${CKPT}"
fi
if [[ "${OUT}" != /* ]]; then
  OUT="${REPO_ROOT}/${OUT}"
fi

# Empty TASKS → lm_eval.sh expands full suite from SUITE.
if [[ -n "${TASKS_OVERRIDE}" ]]; then
  TASKS="${TASKS_OVERRIDE}"
else
  unset TASKS || true
fi
TASKS_DISPLAY="${TASKS:-<suite default: full paper>}"

echo "=== afterok lm_eval ==="
echo "  dep:     afterok:${DEP}"
echo "  ckpt:    ${CKPT} (may be missing until train finishes)"
echo "  out:     ${OUT}"
echo "  profile: ${PROFILE} steps=${STEPS} thr=${THR:-none}"
echo "  stack:   ${STACK} suite=${SUITE} arm=${ARM} tasks=${TASKS_DISPLAY}"

if [[ "${DRY}" -eq 1 ]]; then
  echo "DRY: would sbatch --dependency=afterok:${DEP} lm_eval.sbatch"
  exit 0
fi

# Explicit export list — never inherit parity/UCC pollution from the shell.
# Keep JOB_NAME / intentional knobs; strip only foreign eval overrides.
unset FORCE_DECODE_PROFILE FORCE_UNMASK_THRESHOLD FORCE_GREEDY_PIN FORCE_ARPC \
  ALLOW_FULL_SEQ_DECODE EVAL_DECODE_PROFILE ARPC_OUT HYGIENE_OUT || true
# Drop any ambient TASKS so suite default wins when override unset.
unset TASKS || true
export CKPT OUT_DIR="${OUT}" DECODE_PROFILE="${PROFILE}" NUM_STEPS="${STEPS}"
export SUITE MAX_NEW_TOKENS EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-8192}"
if [[ -n "${TASKS_OVERRIDE}" ]]; then
  export TASKS="${TASKS_OVERRIDE}"
fi
export NUM_NODES GPUS_PER_NODE EVAL_FORWARD FORCE_STACK="${STACK}"
export ALLOW_FULL_SEQ_DECODE=0
export UNMASK_THRESHOLD="${THR}"
export FORCE_GREEDY="${GREEDY_PIN}"
# Slurm job name (do not unset JOB_NAME after this).
export JOB_NAME

jid="$(sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --dependency="afterok:${DEP}" \
  --nodes="${NUM_NODES}" \
  --time="${TIME_LIMIT:-12:00:00}" \
  --export=ALL \
  scripts/slurm/lm_eval.sbatch)"

echo "${jid}" > "${OUT}/submit.jid"
cat > "${OUT}/SUBMITTED.json" <<EOF
{
  "job_id": "${jid}",
  "dependency": "afterok:${DEP}",
  "ckpt": "${CKPT}",
  "profile": "${PROFILE}",
  "num_steps": "${STEPS}",
  "out_dir": "${OUT}",
  "suite": "${SUITE}",
  "tasks": "${TASKS_OVERRIDE:-<suite default>}",
  "submitted_at": "$(date -Iseconds)"
}
EOF
echo "Submitted ${jid} (afterok:${DEP}) → ${OUT}"
