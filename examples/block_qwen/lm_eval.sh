#!/usr/bin/env bash
# Fast-dLLM-style post-train report for a UNI-D² block_qwen checkpoint:
#   1) Task benches (accuracy): MMLU, GPQA, GSM8K, MATH, IFEval,
#      HumanEval(+), MBPP(+)
#   2) tok/s (speed): dedicated BlockSampler throughput + lm-eval generate speed
#
# Usage:
#   bash examples/block_qwen/lm_eval.sh <ckpt>
#   SUITE=core TASKS=gsm8k,ifeval bash examples/block_qwen/lm_eval.sh <ckpt>
#   DECODE_PROFILE=dual_cache UNMASK_THRESHOLD=0.9 FORCE_GREEDY=1 \
#     bash examples/block_qwen/lm_eval.sh <ckpt>
#   SKIP_THROUGHPUT=1 bash examples/block_qwen/lm_eval.sh <ckpt>
#   # Multi-GPU (also set by lm_eval.sbatch):
#   NUM_NODES=1 GPUS_PER_NODE=4 bash examples/block_qwen/lm_eval.sh <ckpt>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

CKPT="${1:?Usage: $0 <checkpoint.ckpt>}"
CKPT="$(readlink -f "${CKPT}")"
OUT_DIR="${OUT_DIR:-$(dirname "$(dirname "${CKPT}")")/lm_eval}"
mkdir -p "${OUT_DIR}"
export CKPT OUT_DIR
# Isolate HF evaluate/metrics caches per job so concurrent lm-evals and
# multi-rank HumanEval/MBPP ``code_eval`` loads do not collide on
# default_experiment-*.arrow (see patch_code_eval_metric_cache).
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-${OUT_DIR}/hf_metrics_cache}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-${OUT_DIR}/hf_evaluate_cache}"
mkdir -p "${HF_METRICS_CACHE}" "${HF_EVALUATE_CACHE}"

# Suites (override with TASKS=...):
#   paper_acc — MMLU + GSM8K + IFEval (paper Table 1 accuracy core; no code)
#   core      — Fast-dLLM v2 eval_script.sh tasks without code
#   code      — HumanEval/MBPP base+plus (lm-eval EvalPlus datasets)
#   fastdllm  — core + code  (full paper columns; code tasks are fragile offline)
SUITE="${SUITE:-fastdllm}"
PAPER_ACC_TASKS="mmlu,gsm8k,ifeval"
CORE_TASKS="mmlu,gpqa_main_n_shot,gsm8k,minerva_math,ifeval"
CODE_TASKS="humaneval,humaneval_plus,mbpp,mbpp_plus"
case "${SUITE}" in
  paper_acc|paper-acc) DEFAULT_TASKS="${PAPER_ACC_TASKS}" ;;
  core) DEFAULT_TASKS="${CORE_TASKS}" ;;
  code) DEFAULT_TASKS="${CODE_TASKS}" ;;
  fastdllm|full|paper) DEFAULT_TASKS="${CORE_TASKS},${CODE_TASKS}" ;;
  custom)
    if [[ -z "${TASKS:-}" ]]; then
      echo "SUITE=custom requires TASKS=..." >&2
      exit 2
    fi
    DEFAULT_TASKS="${TASKS}"
    ;;
  *) echo "Unknown SUITE=${SUITE} (paper_acc|core|code|fastdllm|custom)" >&2; exit 2 ;;
esac
# Empty TASKS (e.g. sbatch export) → suite default.
if [[ -z "${TASKS:-}" ]]; then
  TASKS="${DEFAULT_TASKS}"
fi

# Multi-GPU / multi-node (lm-eval data-parallel via Accelerate, Fast-dLLM-style).
# Under Slurm: NUM_NODES × GPUS_PER_NODE processes, one model replica each.
NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
# Prefer explicit; fall back to Slurm visible device count.
if [[ -z "${GPUS_PER_NODE}" || "${GPUS_PER_NODE}" == "0" ]]; then
  GPUS_PER_NODE=1
fi
# SLURM_GPUS_ON_NODE can be "4(IDX:0-3)" — take leading int.
GPUS_PER_NODE="$(echo "${GPUS_PER_NODE}" | sed -E 's/[^0-9].*//;s/^$/1/')"
NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_NODES * GPUS_PER_NODE))}"
MASTER_PORT="${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-$$} % 1000)))}"
# See lm_eval.sbatch: raise NCCL / c10d timeouts for slow BlockSampler skew.
export LM_EVAL_DIST_TIMEOUT_SEC="${LM_EVAL_DIST_TIMEOUT_SEC:-21600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# Hub Fast-dLLM v2/eval.py default max_new_tokens=2048 (paper accuracy tables).
# Hub Fast-dLLM eval.py defaults to 2048 (context 32k). Our conversion
# models are usually length=2048 — block_qwen_lm_eval clamps max_new so the
# prompt is not wiped (seq_len-max_new must stay >>1). Prefer 512 for fair
# 2k-context roofs unless you know the ckpt context is larger.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_STEPS="${NUM_STEPS:-32}"
LIMIT="${LIMIT:-}"   # e.g. LIMIT=8 for smoke
SKIP_THROUGHPUT="${SKIP_THROUGHPUT:-0}"
# Match Hub eval.py model_args threshold=… / greedy decode.
# Paper §4 accuracy default: threshold=1 (parallel decode off). Hub scripts often 0.9.
UNMASK_THRESHOLD="${UNMASK_THRESHOLD:-}"
FORCE_GREEDY="${FORCE_GREEDY:-}"
# Throughput / generate decode path:
#   baseline     — shared ancestral core for ALL pipelines (clears confidence/ARPC/DualCache)
#   hierarchical — baseline + hierarchical_kv speed knob
#   dual_cache   — Fast-dLLM exactness overlay (masked); pair with UNMASK_THRESHOLD=1 (paper)
#                  or 0.9 (Hub/speed). Pins sub_block_size=8.
#   hubmatch     — Hub use_block_cache=False: hierarchical + single_stream + sub8 + greedy.
# (UNI-D2 ports — NOT Fast-dLLM fused CUDA kernels).
DECODE_PROFILE="${DECODE_PROFILE:-baseline}"

SPEED_PATH="${OUT_DIR}/tok_s_lm_eval.json"
THROUGHPUT_PATH="${OUT_DIR}/tok_s.json"

MODEL_ARGS="checkpoint_path=${CKPT},max_new_tokens=${MAX_NEW_TOKENS},num_steps=${NUM_STEPS},show_speed=True,speed_metrics_path=${SPEED_PATH},decode_profile=${DECODE_PROFILE}"
if [[ -n "${UNMASK_THRESHOLD}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},threshold=${UNMASK_THRESHOLD}"
fi
if [[ -n "${FORCE_GREEDY}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},greedy=${FORCE_GREEDY}"
fi
case "${DECODE_PROFILE}" in
  baseline|hierarchical|hubmatch|dual_cache)
    # Pins applied inside block_qwen_lm_eval via decode_profile=…
    ;;
  *)
    echo "Unknown DECODE_PROFILE=${DECODE_PROFILE} (baseline|hierarchical|hubmatch|dual_cache)" >&2
    exit 2
    ;;
esac

run_python() {
  # Fast-dLLM eval_script.sh uses `accelerate launch` for multi-GPU sharding.
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
  echo "=== accelerate launch processes=${NUM_PROCESSES} nodes=${NUM_NODES} gpus/node=${GPUS_PER_NODE} rank=${machine_rank} master=${main_ip}:${MASTER_PORT} ===" \
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
  local limit_args=()
  case "${task}" in
    mmlu) fewshot_args=(--num_fewshot 5) ;;
    gsm8k|minerva_math) fewshot_args=(--num_fewshot 0) ;;
    # mbpp / mbpp_plus yaml defaults to 3-shot; leave harness default.
  esac
  if [[ -n "${LIMIT}" ]]; then
    limit_args=(--limit "${LIMIT}")
  fi
  echo "=== task=${task} ===" | tee -a "${OUT_DIR}/lm_eval.log"
  run_python -m discrete_diffusion.evaluations.block_qwen_lm_eval \
    --model block_qwen \
    --tasks "${task}" \
    --batch_size "${BATCH_SIZE}" \
    --confirm_run_unsafe_code \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args "${MODEL_ARGS}" \
    --output_path "${OUT_DIR}/${task}" \
    "${fewshot_args[@]}" \
    "${limit_args[@]}" \
    2>&1 | tee -a "${OUT_DIR}/lm_eval.log"
}

IFS=',' read -r -a TASK_ARR <<< "${TASKS}"
for t in "${TASK_ARR[@]}"; do
  t="$(echo "${t}" | xargs)"
  [[ -n "${t}" ]] || continue
  run_task "${t}"
done

# Throughput + SUMMARY only on machine 0 (all nodes join Accelerate task runs).
MACHINE_RANK="${SLURM_NODEID:-0}"
if [[ "${MACHINE_RANK}" != "0" ]]; then
  echo "=== worker node rank=${MACHINE_RANK}: skip throughput/SUMMARY ===" \
    | tee -a "${OUT_DIR}/lm_eval.log"
  exit 0
fi

if [[ "${SKIP_THROUGHPUT}" != "1" ]]; then
  echo "=== decode throughput (tok/s) profile=${DECODE_PROFILE} ===" \
    | tee -a "${OUT_DIR}/lm_eval.log"
  thr_args=()
  if [[ -n "${UNMASK_THRESHOLD}" ]]; then
    thr_args+=("unmask_threshold=${UNMASK_THRESHOLD}")
  fi
  if [[ -n "${FORCE_GREEDY}" ]]; then
    thr_args+=("greedy=${FORCE_GREEDY}")
  fi
  python -u -m discrete_diffusion.evaluations.decode_throughput \
    "checkpoint_path=${CKPT}" \
    "metrics_path=${THROUGHPUT_PATH}" \
    "num_steps=${NUM_STEPS}" \
    "batch_size=${BATCH_SIZE}" \
    "num_batches=${THROUGHPUT_BATCHES:-4}" \
    "warmup_batches=1" \
    "max_new_tokens=${MAX_NEW_TOKENS}" \
    "decode_profile=${DECODE_PROFILE}" \
    mode=conditional \
    device=cuda \
    "${thr_args[@]}" \
    2>&1 | tee -a "${OUT_DIR}/lm_eval.log"
fi

export SUITE DECODE_PROFILE UNMASK_THRESHOLD FORCE_GREEDY MAX_NEW_TOKENS NUM_STEPS
python -u - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
summary = {
    "checkpoint": os.environ.get("CKPT", ""),
    "suite": os.environ.get("SUITE", ""),
    "decode_profile": os.environ.get("DECODE_PROFILE", "baseline"),
    "unmask_threshold": os.environ.get("UNMASK_THRESHOLD") or None,
    "force_greedy": os.environ.get("FORCE_GREEDY") or None,
    "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS") or 0) or None,
    "num_steps": int(os.environ.get("NUM_STEPS") or 0) or None,
    "tasks": {},
    "tok_s_lm_eval": None,
    "tok_s_dedicated": None,
    "paper_parity": {
        "task_accuracy": (
            "lm-eval HumanEval(+)/MBPP(+) use EvalPlus HF datasets "
            "(evalplus/humanevalplus, evalplus/mbppplus) — comparable to "
            "paper Base/Plus columns when prompts/stops match. "
            "Hub-parity gen: max_new_tokens=2048; paper accuracy thr=1; "
            "Hub/speed thr=0.9."),
        "throughput": (
            "tok/s is UNI-D2 BlockSampler; DECODE_PROFILE=dual_cache is "
            "algorithmic DualCache/single_stream — NOT Fast-dLLM fused "
            "CUDA kernels. Do not claim paper tok/s parity."),
    },
}

def pick_score(metrics: dict):
  preferred = (
      "exact_match,flexible-extract",
      "exact_match,strict-match",
      "exact_match,none",
      "acc,none",
      "acc_norm,none",
      "pass@1,none",
      "pass_at_1,none",
      "acc",
      "acc_norm",
      "exact_match",
      "pass@1",
      "pass_at_1",
      "prompt_level_strict_acc,none",
      "inst_level_strict_acc,none",
  )
  for key in preferred:
    if key in metrics and isinstance(metrics[key], (int, float)):
      return float(metrics[key]), key
  for key, val in metrics.items():
    if not isinstance(val, (int, float)):
      continue
    if key.endswith("_stderr") or "stderr" in key:
      continue
    if any(key.startswith(p) for p in (
        "exact_match", "acc", "pass@", "pass_at", "prompt_level", "inst_level")):
      return float(val), key
  return None, None

for p in sorted(out.rglob("results_*.json")):
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    results = data.get("results") or {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        score, score_key = pick_score(metrics)
        summary["tasks"][task] = {
            "score": score,
            "score_key": score_key,
            "metrics": {k: v for k, v in metrics.items()
                        if isinstance(v, (int, float, str, bool))},
            "source": str(p),
        }
for path_key, fname in (
    ("tok_s_lm_eval", "tok_s_lm_eval.json"),
    ("tok_s_dedicated", "tok_s.json"),
):
    fp = out / fname
    if fp.is_file():
        summary[path_key] = json.loads(fp.read_text())
(out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Wrote", out / "SUMMARY.json")
lines = ["# Post-train Fast-dLLM-style report", "",
         f"Checkpoint: `{summary['checkpoint']}`",
         f"Suite: `{summary['suite']}`",
         f"Decode profile: `{summary['decode_profile']}`",
         "",
         "## Task benches (accuracy)", "",
         "| Task | Score | Metric |", "|------|-------|--------|"]
for task, row in sorted(summary["tasks"].items()):
    sc = row.get("score")
    sk = row.get("score_key") or ""
    lines.append(
        f"| {task} | {sc if sc is not None else 'n/a'} | {sk} |")
lines += ["", "## tok/s (speed)", ""]
for label, key in (
    ("During lm-eval generate_until", "tok_s_lm_eval"),
    ("Dedicated decode_throughput", "tok_s_dedicated"),
):
    m = summary.get(key) or {}
    if m:
        lines.append(
            f"- **{label}**: {float(m.get('tok_s', 0)):.2f} tok/s "
            f"({m.get('tokens_generated', '?')} tokens in "
            f"{float(m.get('elapsed_s', 0)):.2f}s; "
            f"profile={m.get('decode_profile', summary['decode_profile'])})"
        )
    else:
        lines.append(f"- **{label}**: (missing)")
pp = summary["paper_parity"]
lines += [
    "",
    "## Paper parity notes",
    "",
    f"- **Accuracy:** {pp['task_accuracy']}",
    f"- **Throughput:** {pp['throughput']}",
    "",
]
(out / "SUMMARY.md").write_text("\n".join(lines))
print("Wrote", out / "SUMMARY.md")
PY

echo "Results under ${OUT_DIR} (see SUMMARY.md)"
