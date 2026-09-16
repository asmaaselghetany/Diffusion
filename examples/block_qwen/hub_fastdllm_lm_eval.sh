#!/usr/bin/env bash
# Official Fast-dLLM v2 Hub model → vendored v2/eval.py (lm-eval).
# Calibrates paper Table 1; does NOT use UNI-D2 BlockSampler.
#
# Usage (usually via scripts/slurm/hub_fastdllm_lm_eval.sbatch):
#   MODEL_PATH=Efficient-Large-Model/Fast_dLLM_v2_1.5B THRESHOLD=1 \
#     bash examples/block_qwen/hub_fastdllm_lm_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FDLLM_ROOT="${REPO_ROOT}/third_party/Fast-dLLM/v2"
cd "${FDLLM_ROOT}"
# Include UNI-D2 src for HumanEval chat postprocess / code_eval cache patches.
export PYTHONPATH="${REPO_ROOT}/src:${FDLLM_ROOT}:${FDLLM_ROOT}/src:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

MODEL_PATH="${MODEL_PATH:-/e/scratch/scifi/elsayed3/hf_cache/models/Fast_dLLM_v2_1.5B}"
THRESHOLD="${THRESHOLD:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
SMALL_BLOCK_SIZE="${SMALL_BLOCK_SIZE:-8}"
BD_SIZE="${BD_SIZE:-32}"
USE_BLOCK_CACHE="${USE_BLOCK_CACHE:-False}"
TASKS="${TASKS:-mmlu,gsm8k,ifeval}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/hub_fastdllm_eval/$(basename "${MODEL_PATH}")_t${THRESHOLD}}"
mkdir -p "${OUT_DIR}"

# Ensure Hub modeling sees local snapshot offline.
if [[ -d "${MODEL_PATH}" ]]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
GPUS_PER_NODE="$(echo "${GPUS_PER_NODE}" | sed -E 's/[^0-9].*//;s/^$/1/')"
NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_NODES * GPUS_PER_NODE))}"
MASTER_PORT="${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-$$} % 1000)))}"
export LM_EVAL_DIST_TIMEOUT_SEC="${LM_EVAL_DIST_TIMEOUT_SEC:-21600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${LM_EVAL_DIST_TIMEOUT_SEC}}"

MODEL_ARGS="model_path=${MODEL_PATH},threshold=${THRESHOLD},max_new_tokens=${MAX_NEW_TOKENS},small_block_size=${SMALL_BLOCK_SIZE},bd_size=${BD_SIZE},use_block_cache=${USE_BLOCK_CACHE},show_speed=True"

echo "=== hub Fast-dLLM lm-eval ===" | tee -a "${OUT_DIR}/lm_eval.log"
echo "  model=${MODEL_PATH} thr=${THRESHOLD} max_new=${MAX_NEW_TOKENS}" | tee -a "${OUT_DIR}/lm_eval.log"
echo "  tasks=${TASKS} out=${OUT_DIR} procs=${NUM_PROCESSES}" | tee -a "${OUT_DIR}/lm_eval.log"

run_python() {
  if [[ "${NUM_PROCESSES}" -le 1 ]]; then
    python -u "$@"
    return
  fi
  local machine_rank="${SLURM_NODEID:-0}"
  local main_ip="${MASTER_ADDR:-}"
  if [[ -z "${main_ip}" && -n "${SLURM_NODELIST:-}" ]]; then
    main_ip="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
  fi
  main_ip="${main_ip:-127.0.0.1}"
  echo "=== accelerate launch processes=${NUM_PROCESSES} rank=${machine_rank} master=${main_ip}:${MASTER_PORT} ===" \
    | tee -a "${OUT_DIR}/lm_eval.log"
  accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_NODES}" \
    --machine_rank "${machine_rank}" \
    --main_process_ip "${main_ip}" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision no \
    "$@"
}

run_task() {
  local task="$1"
  local fewshot_args=()
  local batch
  case "${task}" in
    mmlu) fewshot_args=(--num_fewshot 5); batch="${BATCH_SIZE:-1}" ;;
    gsm8k|minerva_math) fewshot_args=(--num_fewshot 0); batch="${BATCH_SIZE:-32}" ;;
    gpqa_main_n_shot) batch="${BATCH_SIZE:-1}" ;;
    humaneval|humaneval_plus|mbpp|mbpp_plus)
      # Code gens are long; default batch 1 avoids multi-rank memory pressure
      # (prior Hub code job SIGBUS'd at batch=32).
      batch="${BATCH_SIZE:-1}" ;;
    *) batch="${BATCH_SIZE:-32}" ;;
  esac
  echo "=== task=${task} batch=${batch} ===" | tee -a "${OUT_DIR}/lm_eval.log"
  run_python eval.py \
    --tasks "${task}" \
    --batch_size "${batch}" \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args "${MODEL_ARGS}" \
    --output_path "${OUT_DIR}/${task}" \
    "${fewshot_args[@]}" \
    2>&1 | tee -a "${OUT_DIR}/lm_eval.log"
}

IFS=',' read -r -a TASK_ARR <<< "${TASKS}"
for t in "${TASK_ARR[@]}"; do
  t="$(echo "${t}" | xargs)"
  [[ -n "${t}" ]] || continue
  run_task "${t}"
done

# SUMMARY only on machine 0
if [[ "${SLURM_NODEID:-0}" != "0" ]]; then
  echo "=== worker node: skip SUMMARY ===" | tee -a "${OUT_DIR}/lm_eval.log"
  exit 0
fi

export OUT_DIR MODEL_PATH THRESHOLD MAX_NEW_TOKENS USE_BLOCK_CACHE
python -u - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
summary = {
    "model_path": os.environ.get("MODEL_PATH"),
    "threshold": os.environ.get("THRESHOLD"),
    "max_new_tokens": os.environ.get("MAX_NEW_TOKENS"),
    "use_block_cache": os.environ.get("USE_BLOCK_CACHE"),
    "tasks": {},
    "note": (
        "Official Fast-dLLM v2 Hub weights + vendored v2/eval.py "
        "(not UNI-D2 BlockSampler). Paper Table 1 calibration."
    ),
}
for task_dir in sorted(p for p in out.iterdir() if p.is_dir()):
    cands = sorted(task_dir.rglob("results*.json"))
    if not cands:
        continue
    data = json.loads(cands[-1].read_text())
    summary["tasks"][task_dir.name] = data.get("results", {})
(out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Wrote", out / "SUMMARY.json")
PY
