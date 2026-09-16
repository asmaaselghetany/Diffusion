#!/usr/bin/env bash
# Prefetch + submit official Fast-dLLM v2 Hub lm-eval (paper Table 1 calibration).
#
# Uses a dedicated venv (transformers 4.53.1) — train venv is locked at 4.45.
#
# Usage:
#   ./scripts/submit_hub_fastdllm_eval.sh
#   PAPER_BOTH_THR=1 ./scripts/submit_hub_fastdllm_eval.sh
#   PREFETCH_ONLY=1 ./scripts/submit_hub_fastdllm_eval.sh

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

HUB_VENV="${HUB_VENV:-/e/scratch/scifi/elsayed3/.venvs/hub_fastdllm}"
if [[ ! -x "${HUB_VENV}/bin/python" ]]; then
  echo "Missing Hub eval venv: ${HUB_VENV}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${HUB_VENV}/bin/activate"
export PATH="${HUB_VENV}/bin:${PATH}"
export HUB_VENV

# IFEval extras — fail submit early if missing from hub venv.
if ! python -c 'import langdetect, immutabledict, nltk' 2>/dev/null; then
  echo "Hub venv missing IFEval deps (langdetect/immutabledict/nltk)." >&2
  echo "  Fix: pip install -U langdetect immutabledict nltk" >&2
  exit 1
fi
echo "Hub IFEval deps OK"


MODEL_PATH="${MODEL_PATH:-/e/scratch/scifi/elsayed3/hf_cache/models/Fast_dLLM_v2_1.5B}"
if [[ ! -e "${MODEL_PATH}" ]]; then
  MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_1.5B"
fi
THRESHOLD="${THRESHOLD:-1}"
TASKS="${TASKS:-mmlu,gsm8k,ifeval}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

prefetch() {
  echo "=== verify ${MODEL_PATH} ==="
  python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
import jinja2
path = "${MODEL_PATH}"
local = not path.startswith("Efficient-Large-Model/")
# transformers≥4.53 chat templates need jinja2.pass_eval_context (Jinja2≥3.0).
if not hasattr(jinja2, "pass_eval_context"):
  raise SystemExit(
      f"jinja2 {jinja2.__version__} missing pass_eval_context; "
      "pip install 'jinja2>=3.1' in HUB_VENV")
print("Loading", path, "local_files_only=", local)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=local)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, local_files_only=local, torch_dtype="auto")
print("ok", type(model).__name__, "vocab", len(tok), "jinja2", jinja2.__version__)
PY

  # Fast-dLLM eval.py / lm-eval mmlu loads cais/mmlu per subject. Caching only
  # config 'all' is not enough under HF_DATASETS_OFFLINE=1.
  if [[ ",${TASKS}," == *",mmlu,"* ]]; then
    echo "=== ensure cais/mmlu per-subject cache ==="
    # Online only if a subject is missing; otherwise verify offline.
    python - <<'PY'
import os
from pathlib import Path
from datasets import get_dataset_config_names, load_dataset

hf_home = Path(os.environ.get("HF_HOME", ""))
cache_root = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets")) / "cais___mmlu"
disk_cfgs = (
    sorted(p.name for p in cache_root.iterdir() if p.is_dir())
    if cache_root.is_dir() else []
)
# Clear offline flags for optional hub metadata; disk listing is authoritative
# when the full subject tree is already present.
os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
hub_cfgs = []
try:
  hub_cfgs = get_dataset_config_names("cais/mmlu")
except Exception as e:
  print(f"get_dataset_config_names: {e}", flush=True)

# datasets offline stubs sometimes yield only ['default'] — ignore that.
hub_cfgs = [c for c in hub_cfgs if c != "default"]
disk_cfgs = [c for c in disk_cfgs if c != "default"]
if len(disk_cfgs) >= 50:
  cfgs = disk_cfgs
elif hub_cfgs:
  cfgs = hub_cfgs
else:
  cfgs = disk_cfgs
if not cfgs:
  raise SystemExit("no cais/mmlu configs found — cannot prefetch")

def subject_ok(cfg: str) -> bool:
  if cfg == "auxiliary_train":
    return True  # train-only; lm-eval subjects use test
  prev_ds = os.environ.get("HF_DATASETS_OFFLINE")
  prev_hub = os.environ.get("HF_HUB_OFFLINE")
  try:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    load_dataset("cais/mmlu", cfg, split="test")
    return True
  except Exception:
    return False
  finally:
    if prev_ds is None:
      os.environ.pop("HF_DATASETS_OFFLINE", None)
    else:
      os.environ["HF_DATASETS_OFFLINE"] = prev_ds
    if prev_hub is None:
      os.environ.pop("HF_HUB_OFFLINE", None)
    else:
      os.environ["HF_HUB_OFFLINE"] = prev_hub

missing = [c for c in cfgs if not subject_ok(c)]
if missing:
  print(f"missing {len(missing)}/{len(cfgs)} configs — downloading…", flush=True)
  os.environ.pop("HF_DATASETS_OFFLINE", None)
  os.environ.pop("HF_HUB_OFFLINE", None)
  for i, cfg in enumerate(missing):
    print(f"  [{i+1}/{len(missing)}] {cfg}", flush=True)
    load_dataset("cais/mmlu", cfg, trust_remote_code=True)
else:
  print(f"ok offline: {len(cfgs)} cais/mmlu configs", flush=True)

# Final offline gate on a subject that previously failed Hub jobs.
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
n = len(load_dataset("cais/mmlu", "prehistory", split="test"))
print(f"offline gate prehistory test={n}", flush=True)
PY
  fi
}

if [[ "${SKIP_PREFETCH:-0}" != "1" ]]; then
  prefetch
fi
if [[ "${PREFETCH_ONLY:-0}" == "1" ]]; then
  echo "PREFETCH_ONLY=1 — done."
  exit 0
fi

submit_one() {
  local thr="$1"
  local name="$2"
  local out="${OUT_DIR:-${REPO_ROOT}/outputs/hub_fastdllm_eval/$(basename "${MODEL_PATH}")_t${thr}}"
  mkdir -p "${out}" slurm_logs
  MODEL_PATH="${MODEL_PATH}" \
  THRESHOLD="${thr}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}" \
  TASKS="${TASKS}" \
  OUT_DIR="${out}" \
  BATCH_SIZE="${BATCH_SIZE:-}" \
  NUM_NODES="${NUM_NODES}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" \
  NUM_PROCESSES="$((NUM_NODES * GPUS_PER_NODE))" \
  HUB_VENV="${HUB_VENV}" \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  sbatch --parsable \
    --job-name="${name}" \
    --nodes="${NUM_NODES}" \
    --time="${TIME_LIMIT}" \
    --export=ALL \
    scripts/slurm/hub_fastdllm_lm_eval.sbatch
}

echo "=== submit hub Fast-dLLM eval ==="
echo "  model=${MODEL_PATH} tasks=${TASKS} nodes=${NUM_NODES}x${GPUS_PER_NODE}"
echo "  venv=${HUB_VENV}"

id="$(submit_one "${THRESHOLD}" "hub-fdllm-t${THRESHOLD}")"
echo "Submitted thr=${THRESHOLD}: ${id}"

if [[ "${PAPER_BOTH_THR:-0}" == "1" && "${THRESHOLD}" != "0.9" ]]; then
  id09="$(submit_one 0.9 "hub-fdllm-t0.9")"
  echo "Submitted thr=0.9: ${id09}"
fi

squeue -u "${USER}" -o '%.18i %.12P %.28j %.2t %.6D %R' | head -20
