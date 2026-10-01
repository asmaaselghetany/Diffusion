#!/usr/bin/env bash
# Family-aware eval submitter — closes the "which stack?" gap.
#
# Auto-routes from checkpoint hydra config (BASELINE.md contract):
#   masked WITHOUT Hub train/decode pins → conversion_lm_eval
#       = hierarchical (BlockGen packing + conf remask thr=0.9, both arms)
#   masked WITH DualCache/confidence pins → fastdllm_lm_eval
#       = dual_cache lever path (not the conversion baseline)
#   uniform / hybrid / ARPC            → offline ELBO/gen-PPL + ARPC samples
#                                      + generative paper_gen suite
#                                      (mmlu_generative,gsm8k,ifeval)
#                                      NEVER likelihood ``mmlu`` (masked CE)
#                                      U0 floor = hierarchical (same skeleton)
#
# Offline free-gen (via eval_checkpoint → run_block_qwen_eval) always uses
# sample_mode=auto + decode_profile=baseline unless you override those flags.
#
# Usage:
#   ./scripts/submit_family_eval.sh <ckpt>          # hierarchical remask floor for C0/U0
#   FORCE_STACK=fastdllm_lm_eval ...                # DualCache lever ablation
#   FORCE_DECODE_PROFILE=dual_cache ...             # decode-only lever on fixed ckpt
#
# Uniform/hybrid literature notes:
#   BlockGen: generative GSM8K + ELBO/GenPPL (no MMLU loglikelihood).
#   LLaDA-Instruct: conditional generation for MCQ.
#   Duo: principled USDM likelihood bound — not implemented here; do not
#   reuse the masked first-token CE heuristic on uniform.
set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh" 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/e/project1/scifi/elsayed3/Diffusion-new}"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_assert_utils_hash.bash"
assert_forward_process_utils_hash || exit $?
# shellcheck disable=SC1091
source scripts/_infer_block_qwen_eval_profile.bash

DRY_RUN=0
LM_ONLY=0
OFFLINE_ONLY=0
CKPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --lm-eval-only) LM_ONLY=1; shift ;;
    --offline-only) OFFLINE_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    -*)
      echo "Unknown flag: $1" >&2
      exit 2
      ;;
    *)
      CKPT="$1"
      shift
      ;;
  esac
done

[[ -n "${CKPT}" ]] || { echo "Usage: $0 <ckpt|run_dir> [--dry-run]" >&2; exit 2; }
CKPT="$(readlink -f "${CKPT}")"
# Allow a run directory (resolves checkpoints/last.ckpt|best.ckpt).
if [[ -d "${CKPT}" ]]; then
  if [[ -f "${CKPT}/checkpoints/last.ckpt" ]]; then
    CKPT="$(readlink -f "${CKPT}/checkpoints/last.ckpt")"
  elif [[ -f "${CKPT}/checkpoints/best.ckpt" ]]; then
    CKPT="$(readlink -f "${CKPT}/checkpoints/best.ckpt")"
  else
    echo "Run dir has no checkpoints/last.ckpt|best.ckpt: ${CKPT}" >&2
    exit 1
  fi
fi
[[ -f "${CKPT}" ]] || { echo "Missing ckpt: ${CKPT}" >&2; exit 1; }

infer_block_qwen_eval_profile "${CKPT}"
STACK="${FORCE_STACK:-${EVAL_STACK}}"

echo "=== submit_family_eval ==="
echo "  ckpt:    ${CKPT}"
echo "  hydra:   ${EVAL_HYDRA_CFG}"
echo "  forward: ${EVAL_FORWARD}  family=${EVAL_FAMILY}"
echo "  stack:   ${STACK}"
echo "  reason:  ${EVAL_REASON}"

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY: $*"
  else
    "$@"
  fi
}

submit_lm_eval() {
  # Stack defaults are positional. Explicit FORCE_* wins; bare DECODE_PROFILE
  # from a prior shell must not silently demote fastdllm exactness → baseline.
  local profile="${FORCE_DECODE_PROFILE:-$1}"
  local thr="${FORCE_UNMASK_THRESHOLD:-$2}"
  # Prefer explicit FORCE_GREEDY_PIN, then FORCE_GREEDY, then stack default $3.
  local greedy="${FORCE_GREEDY_PIN:-${FORCE_GREEDY:-${3:-}}}"
  export CKPT
  export DECODE_PROFILE="${profile}"
  export UNMASK_THRESHOLD="${thr}"
  export FORCE_GREEDY="${greedy}"
  # Suite: caller must set SUITE for this invocation. Do not silently keep a
  # leftover from a prior cell if the stack has a hard default — stack case
  # arms set SUITE before calling us; here only fill if still empty.
  if [[ -z "${SUITE:-}" ]]; then
    export SUITE=paper_acc
  else
    export SUITE
  fi
  # Propagate arm so lm_eval.sh can refuse likelihood MMLU on uniform/hybrid.
  export EVAL_FORWARD="${EVAL_FORWARD:-}"
  # Hub Fast-dLLM v2/eval.py parity (paper accuracy tables).
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
  # Hub has ~32k context; extend gen buffer beyond train length=2048 when needed.
  export EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-8192}"
  export NUM_STEPS="${NUM_STEPS:-32}"
  # Empty TASKS → lm_eval.sh expands full Fast-dLLM suite from SUITE.
  # Callers may still set TASKS=... for partial / resume runs.
  if [[ -z "${TASKS:-}" ]]; then
    unset TASKS || true
  else
    export TASKS
  fi
  # Tag OUT_DIR so thr/max_new cells do not clobber older lm_eval/ (512) trees.
  if [[ -z "${OUT_DIR:-}" ]]; then
    local run_dir
    run_dir="$(dirname "$(dirname "${CKPT}")")"
    local tag="m${MAX_NEW_TOKENS}_${profile}"
    if [[ -n "${thr}" ]]; then
      tag="${tag}_t${thr}"
    fi
    export OUT_DIR="${run_dir}/lm_eval_${tag}"
  else
    export OUT_DIR
  fi
  export NUM_NODES="${NUM_NODES:-8}"
  export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
  export NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_NODES * GPUS_PER_NODE))}"
  export LM_EVAL_DIST_TIMEOUT_SEC="${LM_EVAL_DIST_TIMEOUT_SEC:-21600}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
  local job_name="${JOB_NAME:-bqwen-lm-eval}"
  echo "  lm_eval: profile=${DECODE_PROFILE} thr=${UNMASK_THRESHOLD:-none} greedy=${FORCE_GREEDY:-task} max_new=${MAX_NEW_TOKENS} suite=${SUITE} tasks=${TASKS:-<suite default>} out=${OUT_DIR} nodes=${NUM_NODES}x${GPUS_PER_NODE}"
  # Never propagate a parent job's /tmp/block_qwen_<jid> into the eval allocation.
  # Also refuse silent full-seq open-loop via leaked ALLOW_FULL_SEQ_DECODE.
  unset ALLOW_FULL_SEQ_DECODE || true
  run env -u TMPDIR -u TEMP -u TMP -u ALLOW_FULL_SEQ_DECODE \
    sbatch --parsable --job-name="${job_name}" --nodes="${NUM_NODES}" --time="${TIME_LIMIT:-12:00:00}" \
    --export=ALL \
    scripts/slurm/lm_eval.sbatch
}

submit_offline() {
  echo "  offline: eval_checkpoint.sbatch"
  run env -u TMPDIR -u TEMP -u TMP \
    sbatch --parsable --job-name="${JOB_NAME:-bqwen-eval}" \
    scripts/slurm/eval_checkpoint.sbatch "${CKPT}"
}

submit_arpc() {
  local run_dir
  run_dir="$(dirname "$(dirname "${CKPT}")")"
  local out="${ARPC_OUT:-${run_dir}/eval_arpc}"
  export ARPC_MODE="${ARPC_MODE:-${EVAL_ARPC_MODE:-blockgen}}"
  # TinyGSM pin — do NOT inherit ckpt-baked divergence from older trains.
  export ARPC_CORRUPTION="${ARPC_CORRUPTION:-ar_metric}"
  export ARPC_AR_METRIC="${ARPC_AR_METRIC:-nll}"
  export ARPC_PREFIX_FILL="${ARPC_PREFIX_FILL:-false}"
  # Default: only run ARPC when the ckpt saw size-1 in train (BlockGen prereq).
  # FORCE_ARPC=1 overrides (decode-only probe; sampler warns if untrained).
  if [[ "${FORCE_ARPC:-0}" != "1" && "${EVAL_HAS_ARPC_SIZE1:-0}" != "1" ]]; then
    echo "  arpc: SKIP (no size-1 in block_size_mixture/block_weights; set FORCE_ARPC=1 to probe anyway)"
    return 0
  fi
  echo "  arpc: out=${out} mode=${ARPC_MODE} corruption=${ARPC_CORRUPTION} size1=${EVAL_HAS_ARPC_SIZE1:-?}"
  run env -u TMPDIR -u TEMP -u TMP \
    sbatch --parsable --job-name="${JOB_NAME:-bqwen-arpc}" --export=ALL \
    scripts/slurm/arpc_decode_eval.sbatch "${CKPT}" "${out}" "${NUM_SAMPLES:-8}"
}

submit_genppl_hygiene() {
  # Unifusion-style (PPL, H) dual + NFE sweep. Default on for uniform/hybrid
  # so GenPPL hygiene is not optional on the U-family.
  if [[ "${SKIP_GENPPL_HYGIENE:-0}" == "1" ]]; then
    echo "  hygiene: SKIP (SKIP_GENPPL_HYGIENE=1)"
    return 0
  fi
  local run_dir
  run_dir="$(dirname "$(dirname "${CKPT}")")"
  local profile="${FORCE_DECODE_PROFILE:-${EVAL_DECODE_PROFILE:-baseline}}"
  # Hygiene uses BlockGen-skeleton baseline coerce (→ hierarchical).
  if [[ "${profile}" == "auto" || "${profile}" == "baseline" ]]; then
    profile=hierarchical
  fi
  local out="${HYGIENE_OUT:-${run_dir}/eval/unifusion_hygiene_${profile}}"
  export CKPT
  export MODE="${HYGIENE_MODE:-all}"
  export DECODE_PROFILE="${profile}"
  export SAMPLE_MODE="${SAMPLE_MODE:-conversion_free}"
  export HYGIENE_OUT="${out}"
  export NUM_STEPS_LIST="${NUM_STEPS_LIST:-8 16 32 64}"
  export RUN_MULTISEED="${RUN_MULTISEED:-0}"
  echo "  hygiene: profile=${DECODE_PROFILE} mode=${MODE} out=${HYGIENE_OUT}"
  run env -u TMPDIR -u TEMP -u TMP \
    sbatch --parsable \
      --job-name="${JOB_NAME:-bqwen-hyg}-genppl" \
      --export=ALL,CKPT,MODE,NUM_STEPS_LIST,NUM_SAMPLES,SAMPLE_MODE,DECODE_PROFILE,RUN_MULTISEED,SEEDS,MAX_NEW_TOKENS,HYGIENE_OUT \
      scripts/slurm/gen_ppl_hygiene.sbatch
}

case "${STACK}" in
  fastdllm_lm_eval)
    if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
      submit_offline
    else
      # Paper §4 accuracy default: DualCache + thr=1 (parallel decode off).
      # Do NOT inherit ckpt-baked thr=0.9 via EVAL_UNMASK_THRESHOLD.
      # Override with FORCE_UNMASK_THRESHOLD=0.9, or PAPER_BOTH_THR=1.
      id="$(submit_lm_eval dual_cache 1 1)"
      echo "Submitted lm_eval job (paper thr): ${id}"
      if [[ "${PAPER_BOTH_THR:-0}" == "1" ]]; then
        _saved_out="${OUT_DIR:-}"
        unset OUT_DIR || true
        JOB_NAME="${JOB_NAME:-bqwen-lm-eval}-thr09" \
          FORCE_UNMASK_THRESHOLD=0.9 \
          id09="$(submit_lm_eval dual_cache 0.9 1)"
        echo "Submitted lm_eval job (Hub thr=0.9): ${id09}"
        if [[ -n "${_saved_out}" ]]; then export OUT_DIR="${_saved_out}"; fi
      fi
      if [[ "${LM_ONLY}" -eq 0 ]]; then
        # Optional offline samples/ELBO alongside (uses ckpt pins).
        oid="$(submit_offline || true)"
        echo "Submitted offline eval job: ${oid}"
      fi
    fi
    ;;
  conversion_lm_eval)
    if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
      submit_offline
    else
      # BASELINE meter: BlockGen packing + conf remask thr=0.9 + greedy.
      # Defer greedy to profile (hierarchical.greedy=true). Never hardcode
      # FORCE_GREEDY=0 here — that silently defeats the remask floor (~56% GSM).
      # hubmatch / DualCache / UCC / ARPC = levers (use FORCE_*).
      id="$(submit_lm_eval hierarchical "" "")"
      echo "Submitted lm_eval job (conversion baseline hierarchical): ${id}"
    fi
    ;;
  blockgen_arpc)
    # BlockGen-protocol offline + ARPC. Optionally skip with --offline-only /
    # SKIP_ARPC=1. Generative chat suite is the Instruct-comparable meter
    # (no masked-CE MMLU). Use --lm-eval-only to skip offline/ARPC.
    if [[ "${LM_ONLY}" -eq 0 ]]; then
      if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
        oid="$(submit_offline)"
        echo "Submitted offline eval job: ${oid}"
      else
        oid="$(submit_offline)"
        echo "Submitted offline eval job: ${oid}"
        if [[ "${SKIP_ARPC:-0}" != "1" ]]; then
          aid="$(submit_arpc)"
          echo "Submitted ARPC decode job: ${aid}"
        fi
      fi
      # Unifusion-style GenPPL hygiene (dual + NFE) — default for uniform.
      hid="$(submit_genppl_hygiene || true)"
      echo "Submitted GenPPL hygiene job: ${hid}"
    fi
    if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
      :
    else
      # Generative suite: same BlockGen skeleton floor as conversion.
      # UCC / quiet / ARPC: FORCE_DECODE_PROFILE=….
      # Never keep a masked leftover (paper_acc/fastdllm) — that wastes an
      # 8-node alloc on the MMLU likelihood refuse path.
      case "${SUITE:-}" in
        paper_gen|paper-gen|uniform_gen) ;;
        *) export SUITE=paper_gen ;;
      esac
      if [[ -z "${TASKS:-}" ]]; then
        unset TASKS || true
      else
        export TASKS
      fi
      _greedy_pin="${FORCE_GREEDY_PIN:-${FORCE_GREEDY:-0}}"
      _u_profile="${FORCE_DECODE_PROFILE:-hierarchical}"
      id="$(submit_lm_eval "${_u_profile}" "" "${_greedy_pin}")"
      echo "Submitted generative lm_eval job (suite=${SUITE} profile=${_u_profile}): ${id}"
    fi
    ;;
  *)
    echo "Unknown EVAL_STACK=${STACK} (set FORCE_STACK=...)" >&2
    exit 2
    ;;
esac
