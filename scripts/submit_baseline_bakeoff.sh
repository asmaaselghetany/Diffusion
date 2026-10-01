#!/usr/bin/env bash
# Phase-0 bake-off: B1–B4 + DualCache (C0) + UCC once (U0 UC only).
#
#   C0 (masked):
#     B1  hierarchical        packing + conf commit thr=0.9 greedy
#     B2  hierarchical_arpc   BlockGen ARPC s=32
#     B3  hierarchical_ss     Hub packing ablation
#     B4  hierarchical_arpc   ARPC s=8
#     DC  dual_cache          DualCache K/V thr=1
#
#   U0 (uniform) — same named profiles; UCC is **only** the UC cell:
#     B1  hierarchical        packing (+ thr pin; greedy forced off on Unif)
#     B2  hierarchical_arpc
#     B3  hierarchical_ss
#     B4  hierarchical_arpc s=8
#     UC  uniform_commit      UCC thr=0.9  ← sole UCC baseline
#     (no U0-DC: DualCache K/V is MASK-only; do not disguise UCC as DC)
#
# Usage:
#   ./scripts/submit_baseline_bakeoff.sh
#   ./scripts/submit_baseline_bakeoff.sh --dry-run
#   TASKS=gsm8k ./scripts/submit_baseline_bakeoff.sh
#   SKIP_EXISTING=1 ./scripts/submit_baseline_bakeoff.sh
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
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

TAG="${BAKEOFF_TAG:-bakeoff_20260924}"
# Full paper suite by default. Override TASKS=gsm8k,ifeval for smoke.
TASKS="${TASKS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_NODES="${NUM_NODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

C0_CKPT="${C0_CKPT:-${REPO_ROOT}/outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt}"
U0_CKPT="${U0_CKPT:-${REPO_ROOT}/outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt}"
[[ -f "${C0_CKPT}" ]] || C0_CKPT="/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt"
[[ -f "${U0_CKPT}" ]] || U0_CKPT="/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt"
C0_CKPT="$(readlink -f "${C0_CKPT}")"
U0_CKPT="$(readlink -f "${U0_CKPT}")"
[[ -f "${C0_CKPT}" ]] || { echo "Missing C0 ckpt" >&2; exit 1; }
[[ -f "${U0_CKPT}" ]] || { echo "Missing U0 ckpt" >&2; exit 1; }

C0_ROOT="$(dirname "$(dirname "${C0_CKPT}")")"
U0_ROOT="$(dirname "$(dirname "${U0_CKPT}")")"

echo "=== baseline bake-off ${TAG} ==="
echo "  C0: ${C0_CKPT}"
echo "  U0: ${U0_CKPT}"
echo "  TASKS=${TASKS} nodes=${NUM_NODES}x${GPUS_PER_NODE}"

submit_one() {
  local arm="$1"
  local cell="$2"
  local profile="$3"
  local steps="$4"
  local greedy_pin="$5"
  local thr="$6"
  local ckpt out stack suite jobname

  if [[ "${arm}" == "C0" ]]; then
    ckpt="${C0_CKPT}"
    out="${C0_ROOT}/lm_eval_${TAG}_${cell}_${profile}_s${steps}"
    stack="conversion_lm_eval"
    suite="${SUITE_MASKED:-paper_acc}"
  else
    ckpt="${U0_CKPT}"
    out="${U0_ROOT}/lm_eval_${TAG}_${cell}_${profile}_s${steps}"
    stack="blockgen_arpc"
    suite="${SUITE_UNIFORM:-paper_gen}"
  fi
  jobname="bake-${arm}-${cell}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out}/SUMMARY.json" ]]; then
    echo "SKIP ${arm}/${cell}: SUMMARY exists at ${out}"
    return 0
  fi
  # Prefer submit.jid / SUBMITTED.json for in-flight; PROTOCOL_LOCK alone can
  # permanently skip a crashed incomplete cell, and misses pending jobs that
  # have not started yet (duplicate 8-node queues).
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
  echo "---- ${arm}/${cell} profile=${profile} steps=${steps} out=${out}"
  if [[ "${DRY}" -eq 1 ]]; then
    ./scripts/submit_family_eval.sh "${ckpt}" --lm-eval-only --dry-run \
      | tee "${out}/submit.log" || true
  else
    ./scripts/submit_family_eval.sh "${ckpt}" --lm-eval-only \
      | tee "${out}/submit.log"
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
    fi
  fi
}

# --- C0 masked ----------------------------------------------------------
submit_one C0 B1 hierarchical      32 1 0.9
submit_one C0 B2 hierarchical_arpc 32 0 ""
submit_one C0 B3 hierarchical_ss   32 1 0.9
submit_one C0 B4 hierarchical_arpc  8 0 ""
submit_one C0 DC dual_cache        32 1 1

# --- U0 uniform: same B1–B4 names; UCC only once (UC) -------------------
# B1/B3 = packing ancestral (no sticky). Conf/UCC = UC only.
# Keep profile names `hierarchical` / `hierarchical_ss` so OUT_DIRs match
# restored ancestral SUMMARYs (~5%) — do NOT switch to hierarchical_ancestral
# (that would mkdir a new path and re-queue a duplicate ~5% job).
submit_one U0 B1 hierarchical      32 0 ""
submit_one U0 B2 hierarchical_arpc 32 0 ""
submit_one U0 B3 hierarchical_ss   32 0 ""
submit_one U0 B4 hierarchical_arpc  8 0 ""
submit_one U0 UC uniform_commit    32 "" ""

echo "=== bake-off submit done ==="
echo "Note: UCC is only U0-UC. No U0-DC (DualCache is MASK-only)."
echo "Note: U0 B1/B3 ancestral SUMMARYs present — SKIP_EXISTING avoids re-queue waste."
