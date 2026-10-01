#!/usr/bin/env bash
# Prefetch lm-eval Hub datasets for offline compute nodes.
# Login node only (HF_HUB_OFFLINE=0).
#
# Usage:
#   ./scripts/prefetch_lm_eval_data.sh
#   SUITE=core ./scripts/prefetch_lm_eval_data.sh
#   TASKS=gsm8k,humaneval_plus,mbpp_plus ./scripts/prefetch_lm_eval_data.sh

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Optional extras used by IFEval / code metrics.
python - <<'PY'
import importlib, subprocess, sys
need = []
for pkg in ('langdetect', 'immutabledict', 'nltk'):
  try:
    importlib.import_module(pkg)
  except ImportError:
    need.append(pkg)
if need:
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *need])
PY

CORE_TASKS="mmlu,gpqa_main_n_shot,gsm8k,minerva_math,ifeval"
CODE_TASKS="humaneval,humaneval_plus,mbpp,mbpp_plus"
PAPER_GEN_TASKS="mmlu_generative,gsm8k,ifeval"
SUITE="${SUITE:-fastdllm}"
case "${SUITE}" in
  core) DEFAULT_TASKS="${CORE_TASKS}" ;;
  code) DEFAULT_TASKS="${CODE_TASKS}" ;;
  paper_gen|paper-gen|uniform_gen) DEFAULT_TASKS="${PAPER_GEN_TASKS}" ;;
  paper_acc|paper-acc) DEFAULT_TASKS="mmlu,gsm8k,ifeval" ;;
  fastdllm|full|paper) DEFAULT_TASKS="${CORE_TASKS},${CODE_TASKS}" ;;
  *) echo "Unknown SUITE=${SUITE}" >&2; exit 2 ;;
esac
if [[ -z "${TASKS:-}" ]]; then
  TASKS="${DEFAULT_TASKS}"
fi
echo "=== Prefetch lm-eval datasets → ${HF_DATASETS_CACHE} ==="
echo "  suite: ${SUITE}"
echo "  tasks: ${TASKS}"

export TASKS
python -u - <<'PY'
import os
from datasets import load_dataset
from lm_eval.tasks import TaskManager

# Belt-and-suspenders HF paths (EvalPlus Base/Plus for paper code columns).
direct = [
    ('gsm8k', 'main'),
    ('openai/openai_humaneval', None),
    ('evalplus/humanevalplus', None),
    ('google-research-datasets/mbpp', 'full'),
    ('evalplus/mbppplus', None),
    ('google/IFEval', None),
    # Official Fast-dLLM / lm-eval mmlu loads cais/mmlu *per subject*
    # (prehistory, …). Caching only name='all' breaks offline Hub eval.
    ('cais/mmlu', 'all'),
    # Generative MMLU (uniform/hybrid paper_gen suite).
    ('hails/mmlu_no_train', 'all'),
    ('Idavidrein/gpqa', 'gpqa_main'),
]
for path, name in direct:
  kw = {'path': path, 'trust_remote_code': True}
  if name:
    kw['name'] = name
  print(f'HF load {path} name={name}', flush=True)
  try:
    ds = load_dataset(**kw)
    print(f'  splits={ {k: len(v) for k, v in ds.items()} }', flush=True)
  except Exception as e:
    print(f'  direct load failed (TaskManager may still work): {e}', flush=True)

# Every cais/mmlu config (subjects + all). Skip train-only auxiliary_train
# test-split checks later; still download the config for completeness.
from datasets import get_dataset_config_names
mmlu_cfgs = get_dataset_config_names('cais/mmlu')
print(f'cais/mmlu configs={len(mmlu_cfgs)}', flush=True)
for i, cfg in enumerate(mmlu_cfgs):
  print(f'HF load cais/mmlu name={cfg} [{i+1}/{len(mmlu_cfgs)}]', flush=True)
  try:
    load_dataset('cais/mmlu', cfg, trust_remote_code=True)
  except Exception as e:
    print(f'  FAIL {cfg}: {e}', flush=True)
    raise

tm = TaskManager()
tasks = [t.strip() for t in os.environ['TASKS'].split(',') if t.strip()]
for name in tasks:
  print(f'TaskManager load {name}', flush=True)
  os.environ['HF_ALLOW_CODE_EVAL'] = '1'
  cfg = tm.load_task_or_group(name)
  print(f'  ok keys={list(cfg) if isinstance(cfg, dict) else type(cfg)}', flush=True)

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
print('Offline reload gsm8k + cais/mmlu:prehistory...', flush=True)
load_dataset('gsm8k', 'main')
n = len(load_dataset('cais/mmlu', 'prehistory', split='test'))
print(f'Offline OK (mmlu_prehistory test={n})', flush=True)
PY

echo "=== Done ==="
du -sh "${HF_HOME}" "${HF_DATASETS_CACHE}" 2>/dev/null || true
