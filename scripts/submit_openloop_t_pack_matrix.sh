#!/usr/bin/env bash
# Tier-2 open-loop T×pack 2×2 + matched ARPC T=0.1 on hard twins.
#
# Priority: hierarchical_arpc_t01 first (matched meter), then ancestral 2×2.
# Ckpts: masked 2020048 ∥ uniform 2020049. Decode-only. No Unif(V) in baseline.
#
# Matrix (ancestral, no ARPC):
#   |              | pack-as-is (ss=F,sub=null)     | ss+sub8                    |
#   | T=1.0        | hierarchical_ancestral (done)  | hierarchical_ss_ancestral   |
#   | T=0.1        | hierarchical_ancestral_t01     | ss_quiet_ancestral         |
#
# Usage:
#   ./scripts/submit_openloop_t_pack_matrix.sh
#   ./scripts/submit_openloop_t_pack_matrix.sh --dry-run
#   SKIP_EXISTING=0 ./scripts/submit_openloop_t_pack_matrix.sh
set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh" 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/e/project1/scifi/elsayed3/Diffusion-new}"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_assert_utils_hash.bash"
assert_forward_process_utils_hash || exit $?
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

TAG="${OPENLOOP_TAG:-openloop_tpack_20261001}"
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

M_CKPT="$(resolve_ckpt \
  "${M_CKPT:-${REPO_ROOT}/outputs/block_qwen/ar2block_masked_2020048/checkpoints/last.ckpt}" \
  "/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_masked_2020048/checkpoints/last.ckpt")"
U_CKPT="$(resolve_ckpt \
  "${U_CKPT:-${REPO_ROOT}/outputs/block_qwen/ar2block_uniform_2020049/checkpoints/last.ckpt}" \
  "/e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_uniform_2020049/checkpoints/last.ckpt")"
M_ROOT="$(dirname "$(dirname "${M_CKPT}")")"
U_ROOT="$(dirname "$(dirname "${U_CKPT}")")"

echo "=== pin diff (Tier-2 2×2 must be T-only / pack-only) ==="
python - <<'PY'
from discrete_diffusion.evaluations.decode_profiles import profile_overrides

def pins(name):
  d = {}
  for tok in profile_overrides(name):
    k, v = tok.split('=', 1)
    d[k.replace('sampling.', '')] = v
  return d

KEYS = (
    'use_arpc', 'unmask_threshold', 'hierarchical_kv', 'use_block_cache',
    'single_stream_decode', 'sub_block_size', 'greedy', 'x0_temperature',
    'arpc_temperature', 'uniform_confidence_sticky',
)
cells = {
    'anc_T1': pins('hierarchical_ancestral'),
    'anc_T01': pins('hierarchical_ancestral_t01'),
    'ss_T1': pins('hierarchical_ss_ancestral'),
    'ss_T01': pins('ss_quiet_ancestral'),
    'arpc_T1': pins('hierarchical_arpc'),
    'arpc_T01': pins('hierarchical_arpc_t01'),
}
for name, p in cells.items():
  print(f"{name}: " + ", ".join(f"{k}={p.get(k,'—')}" for k in KEYS))

def only_t(a, b):
  ka = {k: a[k] for k in a if k != 'x0_temperature' and k != 'arpc_temperature'}
  kb = {k: b[k] for k in b if k != 'x0_temperature' and k != 'arpc_temperature'}
  return ka == kb and a.get('x0_temperature') != b.get('x0_temperature')

assert only_t(cells['anc_T1'], cells['anc_T01']), 'anc row must differ only in T'
assert only_t(cells['ss_T1'], cells['ss_T01']), 'ss row must differ only in T'
# packing column: ss + sub8 flip, T fixed
for top, bot in ((cells['anc_T1'], cells['ss_T1']), (cells['anc_T01'], cells['ss_T01'])):
  assert top['single_stream_decode'] == 'false' and bot['single_stream_decode'] == 'true'
  assert top['sub_block_size'] == 'null' and bot['sub_block_size'] == '8'
  assert top['x0_temperature'] == bot['x0_temperature']
  assert top['use_arpc'] == 'false' and bot['use_arpc'] == 'false'
assert only_t(cells['arpc_T1'], cells['arpc_T01']), 'ARPC T01 must differ only in T(+arpc_T)'
print('PIN_DIFF_OK')
PY

submit_one() {
  local arm="$1"       # M|U
  local cell="$2"
  local profile="$3"
  local steps="$4"
  local ckpt out stack suite jobname root

  if [[ "${arm}" == "M" ]]; then
    ckpt="${M_CKPT}"
    root="${M_ROOT}"
    stack="conversion_lm_eval"
    suite="${SUITE_MASKED:-paper_acc}"
  else
    ckpt="${U_CKPT}"
    root="${U_ROOT}"
    stack="blockgen_arpc"
    suite="${SUITE_UNIFORM:-paper_gen}"
  fi
  out="${root}/lm_eval_${TAG}_${cell}_${profile}_s${steps}"
  jobname="ol-${arm}-${cell}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out}/SUMMARY.json" ]]; then
    echo "SKIP ${arm}/${cell}: SUMMARY exists"
    return 0
  fi
  if [[ "${SKIP_EXISTING}" == "1" && ( -f "${out}/SUBMITTED.json" || -f "${out}/submit.jid" ) ]]; then
    echo "SKIP ${arm}/${cell}: already submitted"
    return 0
  fi

  export CKPT="${ckpt}"
  export FORCE_STACK="${stack}"
  export FORCE_DECODE_PROFILE="${profile}"
  export NUM_STEPS="${steps}"
  export OUT_DIR="${out}"
  if [[ -z "${TASKS:-}" ]]; then unset TASKS || true; else export TASKS; fi
  export SUITE="${suite}"
  export MAX_NEW_TOKENS
  export NUM_NODES
  export GPUS_PER_NODE
  export JOB_NAME="${jobname}"
  export EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-8192}"
  export FORCE_GREEDY_PIN=0
  export FORCE_GREEDY=0
  unset FORCE_UNMASK_THRESHOLD || true

  mkdir -p "${out}"
  echo "---- ${arm}/${cell} profile=${profile} s=${steps} out=${out}"
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
  "cell": "${cell}",
  "tag": "${TAG}",
  "submitted_at": "$(date -Iseconds)"
}
EOF
    fi
  fi
}

echo "=== open-loop T×pack + ARPC-t01  ${TAG} ==="
echo "  M: ${M_CKPT}"
echo "  U: ${U_CKPT}"

# 1) Matched ARPC T=0.1 (higher priority)
submit_one M ARPC_T01 hierarchical_arpc_t01 32
submit_one U ARPC_T01 hierarchical_arpc_t01 32

# 2) Ancestral 2×2 remaining cells (T=1 pack-as-is already done historically)
submit_one M ANC_SS_T1 hierarchical_ss_ancestral 32
submit_one U ANC_SS_T1 hierarchical_ss_ancestral 32
submit_one M ANC_T01 hierarchical_ancestral_t01 32
submit_one U ANC_T01 hierarchical_ancestral_t01 32
submit_one M ANC_SS_T01 ss_quiet_ancestral 32
submit_one U ANC_SS_T01 ss_quiet_ancestral 32

echo "=== done ==="
