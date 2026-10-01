#!/usr/bin/env bash
# Dynamic lever submitter — ONLY way to launch paper-hook micros/runs.
#
# Resolves Hydra overrides from configs/levers/registry.yaml via
# tools/resolve_lever.py. Refuses unknown / conflicting / wrong-arm combos.
#
# Usage:
#   # Paper-scale C2 (6000 / 2048 / 256) — inferred from C2_* / C0 presets
#   ./scripts/submit_lever.sh --preset C2_fdllm --arm masked
#
#   # Explicit micro (500 / 512 / 128)
#   ./scripts/submit_lever.sh --preset C2_fdllm --arm masked --micro
#   # or: MAX_STEPS=500 SEQ_LEN=512 GBS=128 ./scripts/submit_lever.sh ...
#
#   ./scripts/submit_lever.sh --levers shift,complementary --arm masked --micro
#   ./scripts/submit_lever.sh --preset blockgen_uniform --arm uniform --micro
#   ./scripts/submit_lever.sh --list
#
# Never pass raw lever Hydra flags — the registry is the source of truth.
# Extra non-lever overrides go in EXTRA_OVERRIDES.

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
# Login nodes often lack bare `python`; prefer repo venv.
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_assert_utils_hash.bash"
assert_forward_process_utils_hash || exit $?
if ! command -v python >/dev/null 2>&1; then
  echo "No python on PATH (expected ${REPO_ROOT}/.venv/bin/python)" >&2
  exit 1
fi

PRESET=""
LEVERS=""
ARM=""
LINE=""
LINE_SET=0
DRY_RUN=0
LIST=0
SCALE=""   # paper | micro | empty→infer

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="${2:?}"; shift 2 ;;
    --levers) LEVERS="${2:?}"; shift 2 ;;
    --arm) ARM="${2:?}"; shift 2 ;;
    --line) LINE="${2:?}"; LINE_SET=1; shift 2 ;;
    --micro) SCALE=micro; shift ;;
    --paper) SCALE=paper; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST=1; shift ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# If preset locks a single line (e.g. BlockGen scratch), default to it so
# callers cannot silently train ar2block under a BlockGen tag.
if [[ -n "${PRESET}" && "${LINE_SET}" -eq 0 ]]; then
  LINE="$(python - <<PY
import yaml
from pathlib import Path
reg = yaml.safe_load(Path('configs/levers/registry.yaml').read_text())
lines = list((reg.get('presets') or {}).get('${PRESET}', {}).get('line') or [])
print(lines[0] if len(lines) == 1 else 'ar2block')
PY
)"
fi
LINE="${LINE:-ar2block}"

if [[ "${LIST}" -eq 1 ]]; then
  python - <<'PY'
import yaml
from pathlib import Path
reg = yaml.safe_load(Path('configs/levers/registry.yaml').read_text())
print('=== presets ===')
for k, v in (reg.get('presets') or {}).items():
  print(f"  {k}: levers={v.get('levers')} arms={v.get('arms')} line={v.get('line')}")
print('=== levers ===')
for k, v in (reg.get('levers') or {}).items():
  print(f"  {k}: arms={v.get('arms')} :: {v.get('overrides')}")
PY
  exit 0
fi

if [[ -z "${ARM}" ]]; then
  echo "Required: --arm masked|uniform|hybrid" >&2
  exit 1
fi

# Infer scale: paper cells must NOT silently become 500-step micros.
_is_paper_preset() {
  case "${1}" in
    C0|U0|U0_ss_pack|U0_ss_shift|U0_shift|C0_shift|C3_fullseq|C3_fullseq_v2|U2|C2_shift|C2_comp|C2_fdllm|C2_fdllm_full|fastdllm_v2|C5_joint_ar|C5_causal_clean|N0|B3_mixture|B3_arpc|B3_arpc_simplified|B3_t_strat|B3_weights_32|B3_u_stratified|xfer_mixture|xfer_arpc|xfer_weights_32|xfer_u_stratified|xfer_bg_mix_32|xfer_bg_mix_32_blockgen|xfer_bg_mix_32_arpc|xfer_bg_mix_32_arpc_blockgen|xfer_bg_mix_32_blockgen_ss|xfer_bg_mix_32_blockgen_ss_shift|xfer_bg_mix_32_blockgen_unifv|xfer_blockgen_uniform|xfer_blockgen_uniform_32|B4_hybrid_p10|B4_hybrid_p50|B4_joint_curriculum|B4_shift_explorative|C0_anneal|U0_anneal|U0_anneal_shift|decode_sub_block|decode_hierarchical|decode_dual_cache|blockgen_uniform|blockgen_owt_uniform|fdllm|fdllm_decode) return 0 ;;
    *) return 1 ;;
  esac
}
if [[ -z "${SCALE}" ]]; then
  if [[ -n "${PRESET}" ]] && _is_paper_preset "${PRESET}"; then
    SCALE=paper
  else
    SCALE=micro
  fi
fi

# Full-seq diffusion (paper C3): one block = whole sequence.
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
if [[ "${PRESET}" == "C3_fullseq" || "${PRESET}" == C3_fullseq_* ]]; then
  BLOCK="${BLOCK:-2048}"
else
  BLOCK="${BLOCK:-32}"
fi
if [[ "${PRESET}" == "C3_fullseq" || "${PRESET}" == C3_fullseq_* ]] && [[ "${BLOCK}" -ne 2048 ]]; then
  echo "WARN: ${PRESET} expects BLOCK=2048 (got ${BLOCK}); continuing with override." >&2
fi
# Full-seq @ 8×4 ranks: default loader.num_workers=2 blows torch_shm_manager
# (C3_fullseq_v2 job 2125144 exit 143). Successful C3 v1 used num_workers=0.
if [[ "${PRESET}" == "C3_fullseq" || "${PRESET}" == C3_fullseq_* ]]; then
  if [[ " ${EXTRA_OVERRIDES} " != *"loader.num_workers="* ]]; then
    EXTRA_OVERRIDES="${EXTRA_OVERRIDES:+${EXTRA_OVERRIDES} }loader.num_workers=0"
  fi
fi
export NUM_NODES="${NUM_NODES:-1}"
# Jupiter booster nodes are 4×GH200; legacy HFMI default was 2.
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_GPUS="${NUM_GPUS:-$((NUM_NODES * GPUS_PER_NODE))}"
# Booster QOS MaxWall=12h. Paper 6k @ GBS 256: default 8×4 (32 GPUs).
export SBATCH_TIME="${SBATCH_TIME:-12:00:00}"

if [[ "${SCALE}" == "paper" ]]; then
  # Match configs/experiment/block_qwen.yaml — only set knobs if caller overrides.
  MAX_STEPS="${MAX_STEPS:-6000}"
  SEQ_LEN="${SEQ_LEN:-2048}"
  GBS="${GBS:-256}"
  # Default paper width on Jupiter (8 nodes × 4 GPUs).
  if [[ -z "${NUM_NODES_SET:-}" && "${NUM_NODES}" -eq 1 ]]; then
    export NUM_NODES=8
    export NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
  fi
  export RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-false}"
  export RUN_FULL_EVAL="${RUN_FULL_EVAL:-true}"
  export WANDB_PROJECT="${WANDB_PROJECT:-block_qwen}"
  export WANDB_RESUME="${WANDB_RESUME:-never}"
  # t_bucketed optional on full runs (costly); enable via EXTRA if desired.
  BASE_OVERRIDES="trainer.max_steps=${MAX_STEPS} model.length=${SEQ_LEN} block_size=${BLOCK} loader.global_batch_size=${GBS}"
else
  MAX_STEPS="${MAX_STEPS:-500}"
  SEQ_LEN="${SEQ_LEN:-512}"
  GBS="${GBS:-128}"
  export RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-false}"
  export RUN_FULL_EVAL="${RUN_FULL_EVAL:-false}"
  export WANDB_PROJECT="${WANDB_PROJECT:-block_qwen_trials}"
  export WANDB_RESUME="${WANDB_RESUME:-never}"
  BASE_OVERRIDES="trainer.max_steps=${MAX_STEPS} model.length=${SEQ_LEN} block_size=${BLOCK} loader.global_batch_size=${GBS} eval.t_bucketed_nll=true"
fi

resolve_one() {
  local arm="$1"
  local line="$2"
  local args=(--arm "${arm}" --line "${line}" --format json)
  if [[ -n "${PRESET}" ]]; then
    args+=(--preset "${PRESET}")
  else
    args+=(--levers "${LEVERS}")
  fi
  if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    args+=(--extra-overrides "${EXTRA_OVERRIDES}")
  fi

  local meta tag lever_list lever_ov
  meta="$(python tools/resolve_lever.py "${args[@]}")" || {
    echo "resolve_lever failed for arm=${arm} line=${line}" >&2
    exit 2
  }
  tag="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["tag"])' "${meta}")"
  lever_list="$(python -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["levers"]))' "${meta}")"
  lever_ov="$(python -c 'import json,sys; print(" ".join(json.loads(sys.argv[1])["overrides"]))' "${meta}")"
  python -c 'import json,sys
for w in json.loads(sys.argv[1]).get("warnings") or []:
  print("LEVER_WARN:", w, file=sys.stderr)' "${meta}" || true

  export HYDRA_OVERRIDES="${BASE_OVERRIDES} ${lever_ov} ${EXTRA_OVERRIDES}"
  # Filesystem-safe id base (no '+'); clear human title set after job id known.
  # Launch suffixes SLURM_JOB_ID → stable resume key. Display name is rewritten
  # post-train by scripts/rename_wandb_runs.py / upload_wandb_panels.py to:
  #   {Paradigm} · {corruption} · {recipe} · {jobid}
  export WANDB_RUN_NAME="${tag}_${line}_${arm}"
  # Prefer short preset as the visible title seed (avoid lever_shift+comp+…).
  if [[ -n "${PRESET}" ]]; then
    export WANDB_RUN_NAME="${PRESET}_${line}_${arm}"
  fi
  unset WANDB_RUN_ID || true

  # Persist overrides to a unique file before sbatch (job id unknown yet).
  # Avoid last-write-wins on a shared path if another lever is submitted while
  # this job is still pending.
  local ov_dir="${REPO_ROOT}/outputs/block_qwen/lever_overrides"
  mkdir -p "${ov_dir}"
  local ov_stamp
  ov_stamp="$(date +%Y%m%dT%H%M%S)_$$"
  local ov_file="${ov_dir}/${tag}_${line}_${arm}_${ov_stamp}.txt"
  python - "${HYDRA_OVERRIDES}" <<'PY' > "${ov_file}"
import shlex, sys
for tok in shlex.split(sys.argv[1]):
  print(tok)
PY
  # Joined one-liner for launch (HYDRA_OVERRIDES string / LEVER_OVERRIDES_FILE).
  tr '\n' ' ' < "${ov_file}" | sed 's/[[:space:]]*$/\n/' > "${ov_file}.oneline"
  export LEVER_OVERRIDES_FILE="${ov_file}.oneline"
  export HYDRA_OVERRIDES="$(<"${ov_file}.oneline")"

  echo "=== lever submit ==="
  echo "  scale:   ${SCALE}"
  echo "  tag:     ${tag}"
  echo "  levers:  ${lever_list}"
  echo "  line:    ${line}"
  echo "  arm:     ${arm}"
  echo "  wandb:   ${WANDB_PROJECT}"
  echo "  overrides_file: ${LEVER_OVERRIDES_FILE}"
  echo "  overrides: ${HYDRA_OVERRIDES}"

  local sbatch="scripts/slurm/${line}_${arm}.sbatch"
  if [[ ! -f "${sbatch}" ]]; then
    echo "Missing sbatch script: ${sbatch}" >&2
    exit 1
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY_RUN: would sbatch --nodes=${NUM_NODES} --ntasks-per-node=${GPUS_PER_NODE} --gres=gpu:${GPUS_PER_NODE} --time=${SBATCH_TIME} ${sbatch}"
    return 0
  fi
  # Strip eval-only knobs before --export=ALL. Submitting a lever after a
  # parity/UCC/bake shell otherwise poisons train + auto-eval (wrong
  # FORCE_DECODE_PROFILE / OUT_DIR / SUITE / JOB_NAME → clobber + waste).
  unset FORCE_STACK FORCE_DECODE_PROFILE FORCE_UNMASK_THRESHOLD \
    FORCE_GREEDY_PIN FORCE_GREEDY FORCE_ARPC \
    OUT_DIR SUITE TASKS JOB_NAME DECODE_PROFILE UNMASK_THRESHOLD \
    ALLOW_FULL_SEQ_DECODE EVAL_DECODE_PROFILE ARPC_OUT HYGIENE_OUT CKPT \
    || true
  local jid
  jid="$(sbatch --parsable --job-name="lever_${tag}_${arm}" \
    --nodes="${NUM_NODES}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --cpus-per-task=18 \
    --time="${SBATCH_TIME}" \
    --export=ALL \
    "${sbatch}")"
  # Symlink audit name with job id → unique stamp file.
  ln -sfn "$(basename "${ov_file}.oneline")" \
    "${ov_dir}/${tag}_${line}_${arm}_job${jid}.txt"
  # Shared conversion-baseline lock (fingerprint / caps) for all arms.
  local data_env="${ov_dir}/${tag}_${line}_${arm}_job${jid}.data_env.txt"
  python - <<PY > "${data_env}"
from discrete_diffusion.data.conversion_baseline import conversion_baseline_manifest
import json, os
m = conversion_baseline_manifest()
print(f"job: ${jid}")
print(f"preset: ${PRESET}")
print(f"levers: ${lever_list}")
print(f"tag: ${tag}")
print(f"NEMOTRON_SFT_SPLITS={os.environ.get('NEMOTRON_SFT_SPLITS','')}")
print(f"NEMOTRON_SFT_MAX_PER_SPLIT={os.environ.get('NEMOTRON_SFT_MAX_PER_SPLIT','')}")
print(f"DATA_CACHE={os.environ.get('DATA_CACHE','')}")
print(f"RUN_FULL_EVAL={os.environ.get('RUN_FULL_EVAL','')}")
print(f"FORCE_DECODE_PROFILE={os.environ.get('FORCE_DECODE_PROFILE','<unset>')}")
print(f"OUT_DIR={os.environ.get('OUT_DIR','<unset>')}")
print(f"preprocessing: {m['preprocessing_version']}")
print(f"resolved_caps: {json.dumps(m['max_per_split'])}")
print(f"packing: {m['packing']}")
print(f"eval_max_seq_len_default: {m['eval_max_seq_len_default']}")
print("vocab: Hub-keep when train init (see conversion_baseline)")
print("chat: conversion short system")
PY
  echo "Submitted batch job ${jid}"
  echo "  audit: ${ov_dir}/${tag}_${line}_${arm}_job${jid}.txt -> ${LEVER_OVERRIDES_FILE}"
  echo "  data:  ${data_env}"
  return 0
}

case "${LINE}" in
  ar2block|block|blockgen)
    resolve_one "${ARM}" "${LINE}"
    ;;
  *)
    echo "Unknown --line ${LINE} (expected ar2block|block|blockgen)" >&2
    exit 1
    ;;
esac
