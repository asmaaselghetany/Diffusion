#!/usr/bin/env bash
# Same-data masked–uniform parity matrix (Nemotron AR→block).
# BlockGen-layout decode × steps × {C0, U0}; no TinyGSM data; no UCC coerce.
#
# Fills gaps vs Phase-0 bake:
#   quiet T=0.1 ARPC s=8/32 both arms
#   UCC thr=1 (U0)
#   xfer 1855541 (1+32 on Nemotron) × quiet s=8
#
# Usage:
#   ./scripts/submit_parity_matrix.sh
#   ./scripts/submit_parity_matrix.sh --dry-run
#   SKIP_EXISTING=1 TASKS=gsm8k ./scripts/submit_parity_matrix.sh
set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh" 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/e/project1/scifi/elsayed3/Diffusion-new}"
cd "${REPO_ROOT}"

DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

TAG="${PARITY_TAG:-parity_20260925}"
# Full paper suite by default (mmlu[+_generative],gsm8k,ifeval). Override to subset.
TASKS="${TASKS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_NODES="${NUM_NODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

resolve_ckpt() {
  local preferred="$1" fallback="$2"
  if [[ -f "${preferred}" ]]; then
    readlink -f "${preferred}"
  elif [[ -f "${fallback}" ]]; then
    readlink -f "${fallback}"
  else
    echo "Missing ckpt: ${preferred} / ${fallback}" >&2
    exit 1
  fi
}

C0_CKPT="$(resolve_ckpt \
  "${C0_CKPT:-${REPO_ROOT}/outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt}" \
  "/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt")"
U0_CKPT="$(resolve_ckpt \
  "${U0_CKPT:-${REPO_ROOT}/outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt}" \
  "/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt")"
XFER_U_CKPT="$(resolve_ckpt \
  "${XFER_U_CKPT:-${REPO_ROOT}/outputs/block_qwen/ar2block_uniform_1855541/checkpoints/last.ckpt}" \
  "/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_uniform_1855541/checkpoints/last.ckpt")"

C0_ROOT="$(dirname "$(dirname "${C0_CKPT}")")"
U0_ROOT="$(dirname "$(dirname "${U0_CKPT}")")"
XFER_U_ROOT="$(dirname "$(dirname "${XFER_U_CKPT}")")"

echo "=== parity matrix ${TAG} (Nemotron, no TinyGSM data) ==="
echo "  C0:   ${C0_CKPT}"
echo "  U0:   ${U0_CKPT}"
echo "  xfer: ${XFER_U_CKPT}"
echo "  TASKS=${TASKS} nodes=${NUM_NODES}x${GPUS_PER_NODE}"

submit_one() {
  local arm="$1"       # C0|U0|XFER_U
  local cell="$2"
  local profile="$3"
  local steps="$4"
  local greedy_pin="$5"
  local thr="$6"
  local ckpt out stack suite jobname root

  case "${arm}" in
    C0)
      ckpt="${C0_CKPT}"; root="${C0_ROOT}"
      stack="conversion_lm_eval"; suite="${SUITE_MASKED:-paper_acc}"
      ;;
    U0)
      ckpt="${U0_CKPT}"; root="${U0_ROOT}"
      stack="blockgen_arpc"; suite="${SUITE_UNIFORM:-paper_gen}"
      ;;
    XFER_U)
      ckpt="${XFER_U_CKPT}"; root="${XFER_U_ROOT}"
      stack="blockgen_arpc"; suite="${SUITE_UNIFORM:-paper_gen}"
      ;;
    *) echo "bad arm ${arm}" >&2; return 2 ;;
  esac

  out="${root}/lm_eval_${TAG}_${cell}_${profile}_s${steps}"
  if [[ -n "${thr}" && "${thr}" != "null" ]]; then
    out="${out}_t${thr}"
  fi
  jobname="parity-${arm}-${cell}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out}/SUMMARY.json" ]]; then
    echo "SKIP ${arm}/${cell}: SUMMARY exists at ${out}"
    return 0
  fi
  # Pending jobs write submit.jid / SUBMITTED.json at submit time — do NOT
  # skip on PROTOCOL_LOCK alone (that is written only when the batch starts;
  # a crash after lock would permanently skip an incomplete cell). Also do
  # not re-queue when a job is already pending for this OUT_DIR.
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out}/SUBMITTED.json" ]]; then
    echo "SKIP ${arm}/${cell}: SUBMITTED.json exists at ${out}"
    return 0
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out}/submit.jid" ]]; then
    echo "SKIP ${arm}/${cell}: submit.jid exists at ${out}"
    return 0
  fi

  export CKPT="${ckpt}"
  export FORCE_STACK="${stack}"
  export FORCE_DECODE_PROFILE="${profile}"
  export NUM_STEPS="${steps}"
  export OUT_DIR="${out}"
  if [[ -z "${TASKS}" ]]; then unset TASKS || true; else export TASKS; fi
  export SUITE="${suite}"
  export MAX_NEW_TOKENS
  export NUM_NODES
  export GPUS_PER_NODE
  export JOB_NAME="${jobname}"
  export EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-8192}"
  # Size-1-trained xfer: allow ARPC even if Hydra detection is flaky.
  if [[ "${arm}" == "XFER_U" ]]; then
    export FORCE_ARPC=1
    export EVAL_HAS_ARPC_SIZE1=1
  else
    unset FORCE_ARPC || true
    unset EVAL_HAS_ARPC_SIZE1 || true
  fi
  unset FORCE_UNMASK_THRESHOLD || true
  unset FORCE_GREEDY_PIN || true
  unset FORCE_GREEDY || true
  if [[ -n "${thr}" ]]; then
    export FORCE_UNMASK_THRESHOLD="${thr}"
  fi
  if [[ -n "${greedy_pin}" ]]; then
    export FORCE_GREEDY_PIN="${greedy_pin}"
    export FORCE_GREEDY="${greedy_pin}"
  fi

  mkdir -p "${out}"
  echo "---- ${arm}/${cell} profile=${profile} steps=${steps} thr=${thr:-none} out=${out}"
  if [[ "${DRY}" -eq 1 ]]; then
    ./scripts/submit_family_eval.sh "${ckpt}" --lm-eval-only --dry-run \
      | tee "${out}/submit.log" || true
  else
    ./scripts/submit_family_eval.sh "${ckpt}" --lm-eval-only \
      | tee "${out}/submit.log"
    # Capture job id for SKIP_EXISTING before PROTOCOL_LOCK exists.
    local jid
    jid="$(rg -o 'Submitted[^\n]*: ([0-9]+)' -r '$1' "${out}/submit.log" | tail -1 || true)"
    if [[ -z "${jid}" ]]; then
      jid="$(rg -o '^[0-9]+$' "${out}/submit.log" | tail -1 || true)"
    fi
    if [[ -n "${jid}" ]]; then
      echo "${jid}" > "${out}/submit.jid"
      cat > "${out}/SUBMITTED.json" <<EOF
{
  "job_id": "${jid}",
  "ckpt": "${ckpt}",
  "profile": "${profile}",
  "num_steps": "${steps}",
  "out_dir": "${out}",
  "suite": "${suite}",
  "tasks": "${TASKS}",
  "submitted_at": "$(date -Iseconds)"
}
EOF
      echo "  recorded submit.jid=${jid}"
    fi
  fi
}

# --- Quiet ARPC T=0.1 (BlockGen TinyGSM-ish decode; still Nemotron weights) ---
submit_one C0 Q8  hierarchical_quiet  8 0 ""
submit_one C0 Q32 hierarchical_quiet 32 0 ""
submit_one U0 Q8  hierarchical_quiet  8 0 ""
submit_one U0 Q32 hierarchical_quiet 32 0 ""

# --- UCC thr=1 schedule twin (U0 only) ---
submit_one U0 UC1 uniform_commit 32 0 1

# --- Size-1 Nemotron xfer: quiet s=8 (geometry already trained) ---
submit_one XFER_U Q8 hierarchical_quiet 8 0 ""

echo "=== parity matrix submit done ==="
echo "Track: squeue -u \$USER"
echo "Table: docs/research/AR2BLOCK_GSM_TABLE.md"
