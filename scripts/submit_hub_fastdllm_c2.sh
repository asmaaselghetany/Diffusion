#!/usr/bin/env bash
# Hub Fast-dLLM × our C2 mix × Hub eval — external masked-pipeline correctness check.
#
# Steps:
#   1) Ensure train venv (lmflow + deepspeed, no custom CUDA ops)
#   2) Init Fast_dLLM arch from Qwen2.5-1.5B-Instruct weights
#   3) Export C2 Nemotron mix → LMFlow conversation JSON
#   4) Submit Hub LMFlow train (profound / booster)
#   5) Optionally chain Hub lm-eval on the finished run
#
# Usage:
#   ./scripts/submit_hub_fastdllm_c2.sh              # full path
#   DRY_RUN=1 ./scripts/submit_hub_fastdllm_c2.sh    # print plan only
#   SKIP_EXPORT=1 SKIP_INIT=1 ./scripts/submit_hub_fastdllm_c2.sh
#   PREP_ONLY=1 ./scripts/submit_hub_fastdllm_c2.sh  # init+export, no sbatch
#   SUBMIT_EVAL=0 ./scripts/submit_hub_fastdllm_c2.sh

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/jupiter_paths.sh"
activate_jupiter_modules

SCRATCH="${SCRATCH:-/e/scratch/scifi/elsayed3}"
HUB_C2_ROOT="${HUB_C2_ROOT:-${SCRATCH}/hub_fastdllm_c2}"
HUB_TEMPLATE="${HUB_TEMPLATE:-${SCRATCH}/hf_cache/models/Fast_dLLM_v2_1.5B}"
QWEN_PATH="${QWEN_PATH:-${SCRATCH}/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
INIT_DIR="${INIT_DIR:-${HUB_C2_ROOT}/init_qwen15b}"
DATA_DIR="${DATA_DIR:-${HUB_C2_ROOT}/data/c2_mix_hub}"
# Full C2 export lives at data/c2_mix (12G) — Arrow offset-overflows on load.
# Broken naive shards: data/c2_mix_sharded. Use reshape_lmflow_c2_mix.py caps.
HUB_TRAIN_VENV="${HUB_TRAIN_VENV:-${SCRATCH}/.venvs/hub_fastdllm_train}"
HUB_EVAL_VENV="${HUB_VENV:-${SCRATCH}/.venvs/hub_fastdllm}"
FASTDLLM_ROOT="${FASTDLLM_ROOT:-${REPO_ROOT}/third_party/Fast-dLLM/v2}"

export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NEMOTRON_SFT_SPLITS="${NEMOTRON_SFT_SPLITS:-chat,safety,science,math,code}"
export NEMOTRON_SFT_MAX_PER_SPLIT="${NEMOTRON_SFT_MAX_PER_SPLIT:-math=1000000,code=500000}"

DRY_RUN="${DRY_RUN:-0}"
PREP_ONLY="${PREP_ONLY:-0}"
SKIP_INIT="${SKIP_INIT:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_VENV="${SKIP_VENV:-0}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
NUM_NODES="${NUM_NODES:-8}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

mkdir -p "${HUB_C2_ROOT}/data" "${HUB_C2_ROOT}/output" slurm_logs

echo "=== Hub Fast-dLLM × C2 mix ==="
echo "  root=${HUB_C2_ROOT}"
echo "  init=${INIT_DIR}"
echo "  data=${DATA_DIR}"
echo "  train_venv=${HUB_TRAIN_VENV}"
echo "  splits=${NEMOTRON_SFT_SPLITS} caps=${NEMOTRON_SFT_MAX_PER_SPLIT}"

ensure_train_venv() {
  if [[ "${SKIP_VENV}" == "1" ]]; then
    return 0
  fi
  # Prefer the Hub CUDA eval venv (torch 2.6+cu126 + transformers 4.53.1).
  # Train extras (deepspeed) are installed --no-deps into that same env; lmflow
  # is imported via PYTHONPATH to third_party/Fast-dLLM/third_party.
  if [[ ! -x "${HUB_EVAL_VENV}/bin/python" ]]; then
    echo "Missing Hub eval venv at ${HUB_EVAL_VENV}" >&2
    exit 1
  fi
  if [[ "${HUB_TRAIN_VENV}" != "${HUB_EVAL_VENV}" ]]; then
    if [[ -L "${HUB_TRAIN_VENV}" || ! -e "${HUB_TRAIN_VENV}" ]]; then
      ln -sfn "${HUB_EVAL_VENV}" "${HUB_TRAIN_VENV}"
    fi
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: train venv → ${HUB_TRAIN_VENV}"
    return 0
  fi
  export CUDA_HOME="${CUDA_HOME:-/e/software/default/stages/2026/software/CUDA/13}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  if ! "${HUB_TRAIN_VENV}/bin/python" -c 'import deepspeed' 2>/dev/null; then
    echo "=== install deepspeed (no CUDA ops) into ${HUB_TRAIN_VENV} ==="
    DS_BUILD_OPS=0 "${HUB_TRAIN_VENV}/bin/pip" install --no-deps 'deepspeed>=0.14.4'
    "${HUB_TRAIN_VENV}/bin/pip" install --no-deps hjson msgpack ninja py-cpuinfo
    "${HUB_TRAIN_VENV}/bin/python" -c 'import pydantic' 2>/dev/null \
      || "${HUB_TRAIN_VENV}/bin/pip" install 'pydantic>=2'
  fi
  PYTHONPATH="${FASTDLLM_ROOT}/../third_party:${PYTHONPATH:-}" \
    "${HUB_TRAIN_VENV}/bin/python" - <<'PY'
import torch, deepspeed, lmflow, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("tf", transformers.__version__, "ds", deepspeed.__version__)
print("lmflow", lmflow.__file__)
if not torch.version.cuda:
  raise SystemExit("FATAL: CPU torch in train venv")
try:
  import triton
  print("triton", triton.__version__)
except ImportError:
  print("triton: MISSING (ok on aarch64 — train sbatch disables dynamo)")
PY
}

run_init() {
  if [[ "${SKIP_INIT}" == "1" ]]; then
    echo "SKIP_INIT=1"
    return 0
  fi
  if [[ -f "${INIT_DIR}/INIT_FROM_QWEN.json" && "${FORCE_INIT:-0}" != "1" ]]; then
    echo "init exists: ${INIT_DIR}"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: init_fastdllm_from_qwen.py → ${INIT_DIR}"
    return 0
  fi
  "${HUB_EVAL_VENV}/bin/python" tools/init_fastdllm_from_qwen.py \
    --hub-template "${HUB_TEMPLATE}" \
    --qwen "${QWEN_PATH}" \
    --out "${INIT_DIR}" \
    --force
}

run_export() {
  if [[ "${SKIP_EXPORT}" == "1" ]]; then
    echo "SKIP_EXPORT=1"
    return 0
  fi
  if compgen -G "${DATA_DIR}/train_*.json" >/dev/null && [[ "${FORCE_EXPORT:-0}" != "1" ]]; then
    echo "export exists under ${DATA_DIR}"
    ls -lh "${DATA_DIR}"/train_*.json | head
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: export_c2_mix_to_lmflow.py --dry-run"
    PYTHONPATH=src \
      HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
      HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      "${REPO_ROOT}/.venv/bin/python" tools/export_c2_mix_to_lmflow.py \
        --out-dir "${DATA_DIR}" --dry-run
    return 0
  fi
  echo "=== export C2 mix (long) ==="
  PYTHONPATH=src \
    HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
    "${REPO_ROOT}/.venv/bin/python" tools/export_c2_mix_to_lmflow.py \
      --out-dir "${DATA_DIR}"
}

submit_train() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: sbatch hub_fastdllm_c2_train.sbatch nodes=${NUM_NODES} resume=${RESUME_FROM_CHECKPOINT:-} out=${OUTPUT_DIR:-}"
    return 0
  fi
  mkdir -p slurm_logs
  # Optional resume: OUTPUT_DIR + RESUME_FROM_CHECKPOINT=auto|/path/checkpoint-N
  export HUB_C2_ROOT MODEL_PATH="${INIT_DIR}" DATASET_PATH="${DATA_DIR}"
  export HUB_TRAIN_VENV NUM_NODES GPUS_PER_NODE=4
  [[ -n "${OUTPUT_DIR:-}" ]] && export OUTPUT_DIR
  [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]] && export RESUME_FROM_CHECKPOINT
  [[ -n "${MAX_STEPS:-}" ]] && export MAX_STEPS
  local tid
  tid="$(sbatch --parsable \
    --account="${JUWELS_ACCOUNT:-profound}" \
    --nodes="${NUM_NODES}" \
    --time="${TIME_LIMIT}" \
    --export=ALL \
    scripts/slurm/hub_fastdllm_c2_train.sbatch)"
  echo "Submitted train: ${tid}"
  TRAIN_JOB_ID="${tid}"
}

submit_eval() {
  if [[ "${SUBMIT_EVAL}" != "1" ]]; then
    echo "SUBMIT_EVAL=0 — skip Hub eval chain"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: would chain Hub lm_eval afterok train"
    return 0
  fi
  if [[ -z "${TRAIN_JOB_ID:-}" ]]; then
    echo "No TRAIN_JOB_ID — skip eval chain" >&2
    return 1
  fi
  local out_eval="${HUB_C2_ROOT}/eval/c2_qwen_init_t1"
  mkdir -p "${out_eval}"
  # Train job writes ${HUB_C2_ROOT}/output/latest → run dir.
  # IMPORTANT: do NOT put comma-lists in sbatch --export=ALL,KEY=a,b
  # (Slurm splits on commas). Export into the environment, then --export=ALL.
  export MODEL_PATH="${HUB_C2_ROOT}/output/latest"
  export OUT_DIR="${out_eval}"
  export THRESHOLD="${THRESHOLD:-1}"
  export TASKS="${TASKS:-mmlu,gsm8k,ifeval}"
  export HUB_VENV="${HUB_EVAL_VENV}"
  export NUM_NODES=4
  export GPUS_PER_NODE=4
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  local eid
  eid="$(sbatch --parsable \
    --account="${JUWELS_ACCOUNT:-profound}" \
    --dependency=afterok:"${TRAIN_JOB_ID}" \
    --job-name=hub-fdllm-c2-ev \
    --nodes=4 \
    --time=12:00:00 \
    --export=ALL \
    scripts/slurm/hub_fastdllm_lm_eval.sbatch)"
  echo "Submitted eval (afterok:${TRAIN_JOB_ID}): ${eid}"
  EVAL_JOB_ID="${eid}"
}

ensure_train_venv
run_init
run_export

if [[ "${PREP_ONLY}" == "1" ]]; then
  echo "PREP_ONLY=1 — done (no sbatch)."
  exit 0
fi

submit_train
submit_eval

echo "=== queued ==="
squeue -u "${USER}" -o '%.10i %.28j %.10T %.6D %R' | head -20
