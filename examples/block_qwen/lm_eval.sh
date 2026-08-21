#!/usr/bin/env bash
# Fast-dLLM-style post-train report for a UNI-D² block_qwen checkpoint:
#   1) Task benches (accuracy): MMLU, GPQA, GSM8K, MATH, IFEval, HumanEval
#   2) tok/s (speed): dedicated BlockSampler throughput + lm-eval generate speed
#
# Usage:
#   bash examples/block_qwen/lm_eval.sh \
#     outputs/block_qwen/ar2block_masked_139760/checkpoints/last.ckpt
#   TASKS=gsm8k,ifeval,humaneval bash examples/block_qwen/lm_eval.sh <ckpt>
#   SKIP_THROUGHPUT=1 bash examples/block_qwen/lm_eval.sh <ckpt>
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

# Fast-dLLM v2 suite + HumanEval (code). MBPP/EvalPlus optional via TASKS=.
TASKS="${TASKS:-mmlu,gpqa_main_n_shot,gsm8k,minerva_math,ifeval,humaneval}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NUM_STEPS="${NUM_STEPS:-32}"
LIMIT="${LIMIT:-}"   # e.g. LIMIT=8 for smoke
SKIP_THROUGHPUT="${SKIP_THROUGHPUT:-0}"

SPEED_PATH="${OUT_DIR}/tok_s_lm_eval.json"
THROUGHPUT_PATH="${OUT_DIR}/tok_s.json"
SUMMARY_PATH="${OUT_DIR}/SUMMARY.json"

MODEL_ARGS="checkpoint_path=${CKPT},max_new_tokens=${MAX_NEW_TOKENS},num_steps=${NUM_STEPS},show_speed=True,speed_metrics_path=${SPEED_PATH}"

run_task() {
  local task="$1"
  local fewshot_args=()
  local limit_args=()
  case "${task}" in
    mmlu) fewshot_args=(--num_fewshot 5) ;;
    gsm8k|minerva_math) fewshot_args=(--num_fewshot 0) ;;
  esac
  if [[ -n "${LIMIT}" ]]; then
    limit_args=(--limit "${LIMIT}")
  fi
  echo "=== task=${task} ===" | tee -a "${OUT_DIR}/lm_eval.log"
  python -u -m discrete_diffusion.evaluations.block_qwen_lm_eval \
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

if [[ "${SKIP_THROUGHPUT}" != "1" ]]; then
  echo "=== decode throughput (tok/s) ===" | tee -a "${OUT_DIR}/lm_eval.log"
  python -u -m discrete_diffusion.evaluations.decode_throughput \
    "checkpoint_path=${CKPT}" \
    "metrics_path=${THROUGHPUT_PATH}" \
    "num_steps=${NUM_STEPS}" \
    "batch_size=${BATCH_SIZE}" \
    "num_batches=${THROUGHPUT_BATCHES:-4}" \
    "warmup_batches=1" \
    "max_new_tokens=${MAX_NEW_TOKENS}" \
    mode=conditional \
    device=cuda \
    2>&1 | tee -a "${OUT_DIR}/lm_eval.log"
fi

python -u - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
summary = {
    "checkpoint": os.environ.get("CKPT", ""),
    "tasks": {},
    "tok_s_lm_eval": None,
    "tok_s_dedicated": None,
}

def pick_score(metrics: dict):
  preferred = (
      "exact_match,flexible-extract",
      "exact_match,strict-match",
      "exact_match,none",
      "acc,none",
      "acc_norm,none",
      "pass@1,none",
      "acc",
      "acc_norm",
      "exact_match",
      "pass@1",
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
    if any(key.startswith(p) for p in ("exact_match", "acc", "pass@", "prompt_level", "inst_level")):
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
         f"Checkpoint: `{summary['checkpoint']}`", "",
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
            f"{float(m.get('elapsed_s', 0)):.2f}s)"
        )
    else:
        lines.append(f"- **{label}**: (missing)")
lines += [
    "",
    "_Note: tok/s is UNI-D² BlockSampler (no Fast-dLLM hierarchical KV / "
    "sub-block parallel)._",
    "",
]
(out / "SUMMARY.md").write_text("\n".join(lines))
print("Wrote", out / "SUMMARY.md")
PY

echo "Results under ${OUT_DIR} (see SUMMARY.md)"
