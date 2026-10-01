#!/usr/bin/env bash
#SBATCH --job-name=bqwen-lm-eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Conditional lm-eval + optional tok/s for one checkpoint.
# Multi-GPU / multi-node via Accelerate (Fast-dLLM eval.py style):
#   one model replica per GPU; lm-eval shards examples across ranks.
# Default: 8 nodes × 4 GPUs = 32 processes.
#
# Usage:
#   CKPT=outputs/.../last.ckpt sbatch scripts/slurm/lm_eval.sbatch
#   # 1 node × 4 GPUs:
#   NUM_NODES=1 CKPT=... sbatch --nodes=1 scripts/slurm/lm_eval.sbatch
#   # 1 GPU smoke:
#   NUM_NODES=1 GPUS_PER_NODE=1 CKPT=... sbatch --nodes=1 --gres=gpu:1 \
#     --cpus-per-task=18 scripts/slurm/lm_eval.sbatch
#   DECODE_PROFILE=dual_cache UNMASK_THRESHOLD=0.9 FORCE_GREEDY=1 \
#     CKPT=... sbatch scripts/slurm/lm_eval.sbatch
#
# Prefetch on login node first: ./scripts/prefetch_lm_eval_data.sh
#
# Do NOT pass comma-lists via sbatch --export=...,TASKS=a,b — Slurm splits
# on commas. Set TASKS inside an env file or use SUITE=.

set -euo pipefail

export ASMAA_WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

# Compute nodes have no Hub; require login-node prefetch.
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

CKPT="${CKPT:?Set CKPT=/path/to.ckpt}"
CKPT="$(readlink -f "${CKPT}")"
[[ -f "${CKPT}" ]] || { echo "Missing ${CKPT}" >&2; exit 1; }

export SUITE="${SUITE:-fastdllm}"
export TASKS="${TASKS:-}"
# Hub eval.py default; task ceilings in cap_max_new_for_task still apply
# (mmlu_generative→64) so paper_gen finishes in 12h.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
# Allow lm-eval to extend buffer past train length (Hub max_new=2048 parity).
export EVAL_MAX_SEQ_LEN="${EVAL_MAX_SEQ_LEN:-8192}"
export NUM_STEPS="${NUM_STEPS:-32}"
export SKIP_THROUGHPUT="${SKIP_THROUGHPUT:-0}"
export UNMASK_THRESHOLD="${UNMASK_THRESHOLD:-}"
export FORCE_GREEDY="${FORCE_GREEDY:-}"
export DECODE_PROFILE="${DECODE_PROFILE:-baseline}"
# Optional OUT_DIR so thr=1 vs thr=0.9 / old 512 runs do not clobber.
export OUT_DIR="${OUT_DIR:-}"

# Parallelism: sbatch --nodes must match NUM_NODES (default 8).
export NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-8}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_NODES * GPUS_PER_NODE))}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + (SLURM_JOB_ID % 1000)))}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)}"

# BlockSampler decode is slow; finished ranks enter lm-eval's
# accelerator.wait_for_everyone() while stragglers still generate.
# Default NCCL watchdog (600s) aborted job 1694730 mid-GSM8K gather.
# Raise both the c10d process-group timeout and the NCCL heartbeat.
export LM_EVAL_DIST_TIMEOUT_SEC="${LM_EVAL_DIST_TIMEOUT_SEC:-21600}"  # 6h
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${LM_EVAL_DIST_TIMEOUT_SEC}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${LM_EVAL_DIST_TIMEOUT_SEC}}"

# Resolve EVAL_FORWARD early (needed for uniform protocol lock).
if [[ -z "${EVAL_FORWARD:-}" ]]; then
  # shellcheck disable=SC1091
  source scripts/_infer_block_qwen_eval_profile.bash
  infer_block_qwen_eval_profile "${CKPT}"
fi

# DECODE_PROFILE=auto → DualCache exactness vs baseline from run hydra cfg.
# Prefer ./scripts/submit_family_eval.sh for routing (also refuses uniform).
if [[ "${DECODE_PROFILE}" == "auto" ]]; then
  # shellcheck disable=SC1091
  source scripts/_infer_block_qwen_eval_profile.bash
  infer_block_qwen_eval_profile "${CKPT}"
  case "${EVAL_STACK}" in
    fastdllm_lm_eval)
      export DECODE_PROFILE=dual_cache
      # Paper §4 accuracy default thr=1; Hub/speed often 0.9.
      # Prefer explicit UNMASK_THRESHOLD; else paper thr=1 (ignore ckpt 0.9).
      export UNMASK_THRESHOLD="${UNMASK_THRESHOLD:-1}"
      export FORCE_GREEDY="${FORCE_GREEDY:-1}"
      ;;
    conversion_lm_eval)
      # BlockGen packing + conf remask floor (both arms): hierarchical thr=0.9.
      export DECODE_PROFILE=hierarchical
      if [[ "${EVAL_FORWARD:-}" == "uniform" || "${EVAL_FORWARD:-}" == "hybrid" ]]; then
        export FORCE_GREEDY=0
      fi
      ;;
    blockgen_arpc)
      # Generative-only path allowed when suite avoids likelihood MMLU.
      if [[ "${SUITE:-}" == "paper_acc" || "${SUITE:-}" == "paper-acc" || ",${TASKS:-}," == *",mmlu,"* ]]; then
        echo "REFUSED: likelihood MMLU on uniform/hybrid — use SUITE=paper_gen" >&2
        echo "Or: ./scripts/submit_family_eval.sh <ckpt>  # routes blockgen_arpc" >&2
        exit 2
      fi
      export DECODE_PROFILE=hierarchical
      export FORCE_GREEDY=0
      ;;
    *)
      echo "DECODE_PROFILE=auto could not classify ckpt; set DECODE_PROFILE explicitly" >&2
      exit 2
      ;;
  esac
  echo "  auto-decode: stack=${EVAL_STACK} → profile=${DECODE_PROFILE} thr=${UNMASK_THRESHOLD:-none}"
fi

# ``baseline`` → hierarchical (BlockGen block skeleton). Full-seq only via
# ALLOW_FULL_SEQ_DECODE=1.
if [[ "${DECODE_PROFILE}" == "baseline" || "${DECODE_PROFILE}" == "auto" ]]; then
  if [[ "${ALLOW_FULL_SEQ_DECODE:-0}" == "1" ]]; then
    echo "WARNING: DECODE_PROFILE=baseline with ALLOW_FULL_SEQ_DECODE=1 (legacy full-seq ablation)." >&2
  else
    echo "NOTE: DECODE_PROFILE=baseline → hierarchical (thr=0.9 greedy remask floor; ancestral = hierarchical_ancestral)." >&2
    export DECODE_PROFILE=hierarchical
    if [[ "${EVAL_FORWARD:-}" == "uniform" || "${EVAL_FORWARD:-}" == "hybrid" ]]; then
      export FORCE_GREEDY="${FORCE_GREEDY:-0}"
    fi
  fi
fi
if [[ "${EVAL_FORWARD:-}" == "uniform" || "${EVAL_FORWARD:-}" == "hybrid" ]]; then
  if [[ "${FORCE_GREEDY}" == "1" || "${FORCE_GREEDY}" == "true" || "${FORCE_GREEDY}" == "True" ]]; then
    echo "REFUSED: FORCE_GREEDY=${FORCE_GREEDY} is illegal for forward=${EVAL_FORWARD}." >&2
    echo "  argmax(q_xs) locks prior Unif noise. Use FORCE_GREEDY=0." >&2
    exit 2
  fi
  export FORCE_GREEDY="${FORCE_GREEDY:-0}"
fi

echo "=== lm_eval ==="
echo "  ckpt:   ${CKPT}"
echo "  suite:  ${SUITE}  tasks: ${TASKS:-<suite default>}"
echo "  steps:  ${NUM_STEPS}"
echo "  decode: profile=${DECODE_PROFILE} threshold=${UNMASK_THRESHOLD:-ckpt-default} greedy=${FORCE_GREEDY:-task-default}"
echo "  forward: ${EVAL_FORWARD:-unknown}"
echo "  parallel: nodes=${NUM_NODES} gpus/node=${GPUS_PER_NODE} processes=${NUM_PROCESSES}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  dist timeout: LM_EVAL_DIST_TIMEOUT_SEC=${LM_EVAL_DIST_TIMEOUT_SEC}s (NCCL heartbeat matched)"
echo "  HF_HOME=${HF_HOME} offline=${HF_HUB_OFFLINE}"
echo "  OUT_DIR=${OUT_DIR:-<default beside ckpt>}"

# Forensic lockfile — proves protocol before any generation.
if [[ -n "${OUT_DIR:-}" ]]; then
  mkdir -p "${OUT_DIR}"
  cat > "${OUT_DIR}/PROTOCOL_LOCK.json" <<EOF
{
  "ckpt": "${CKPT}",
  "decode_profile": "${DECODE_PROFILE}",
  "force_greedy": "${FORCE_GREEDY:-}",
  "unmask_threshold": "${UNMASK_THRESHOLD:-}",
  "eval_forward": "${EVAL_FORWARD:-}",
  "suite": "${SUITE}",
  "tasks": "${TASKS:-}",
  "num_steps": "${NUM_STEPS}",
  "max_new_tokens": "${MAX_NEW_TOKENS}",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "account": "${SLURM_JOB_ACCOUNT:-}"
}
EOF
  echo "  wrote ${OUT_DIR}/PROTOCOL_LOCK.json"
fi

# Fail fast on login-visible mistakes before burning an 8-node allocation.
# (Runs on the batch script host = first allocated node after start.)
preflight_tasks="${TASKS:-}"
if [[ -z "${preflight_tasks}" ]]; then
  case "${SUITE}" in
    paper_gen|paper-gen|uniform_gen) preflight_tasks="mmlu_generative,gsm8k,ifeval" ;;
    paper_acc|paper-acc) preflight_tasks="mmlu,gsm8k,ifeval" ;;
    *) preflight_tasks="" ;;
  esac
fi
if [[ -n "${preflight_tasks}" ]]; then
  echo "=== preflight offline task load (${preflight_tasks}) ==="
  python - <<PY || { echo "PREFLIGHT FAILED: datasets not offline-ready under HF_HOME=${HF_HOME}" >&2; exit 2; }
import os
from discrete_diffusion.evaluations.block_qwen_eval_utils import cap_max_new_for_task
from lm_eval.tasks import TaskManager
assert cap_max_new_for_task("mmlu_generative", 2048) == 64, "max_new cap missing"
tm = TaskManager()
for t in "${preflight_tasks}".split(","):
    t = t.strip()
    if not t:
        continue
    tm.load_task_or_group(t)
    print(f"  OK {t} (cap→{cap_max_new_for_task(t, int(os.environ.get('MAX_NEW_TOKENS', '2048')))})")
print("preflight OK")
PY
fi

# Jupiter ParaStation: first multi-node srun often fails (PSI doSpawn).
# shellcheck disable=SC1091
source scripts/_srun_retry.bash
srun_warmup

# One Accelerate launcher per node (spawns GPUS_PER_NODE workers locally).
srun_retry --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
  --cpu-bind=none \
  bash examples/block_qwen/lm_eval.sh "${CKPT}"

# Active WandB upload of lm_eval + offline metrics (best-effort).
if [[ "${WANDB_UPLOAD_EVAL:-1}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  _run_root="$(dirname "$(dirname "$(readlink -f "${CKPT}")")")"
  _jid="$(basename "${_run_root}" | grep -oE '[0-9]+$' || true)"
  if [[ -n "${_jid}" ]]; then
    echo "=== wandb attach eval → train run (job=${_jid}) ==="
    JOBS="${_jid}" SKIP_SCOREBOARD=1 python -u scripts/upload_wandb_panels.py \
      || echo "WARNING: wandb eval upload failed (non-fatal)" >&2
  fi
fi
