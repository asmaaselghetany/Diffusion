#!/bin/bash
#SBATCH --job-name=ced_eval_caps
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_eval_caps_%j.out
#SBATCH --error=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_eval_caps_%j.err

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/continuous_embedding/eval/capability}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/continuous_embedding}"
PRESET="${PRESET:-quick}"           # quick | full
TASK="${TASK:-all}"                 # reconstruction | unconditional | infill | prompt_qa | all
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${REPO_ROOT}/outputs/continuous_embedding/logs"

source ~/.bashrc || true
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Resolve a Python executable with required dependencies.
# Prefer known conda envs via `conda run` to avoid non-interactive shell
# activation issues on SLURM workers.
PYTHON_CMD=(python)
if command -v conda >/dev/null 2>&1; then
  if conda run -n discrete_diffusion python -c "import numpy, torch, omegaconf" >/dev/null 2>&1; then
    PYTHON_CMD=(conda run -n discrete_diffusion python)
  elif conda run -n text-diffusion-jepa python -c "import numpy, torch, omegaconf" >/dev/null 2>&1; then
    PYTHON_CMD=(conda run -n text-diffusion-jepa python)
  fi
fi

if [[ "${PYTHON_CMD[*]}" == "python" ]]; then
  source venv/bin/activate 2>/dev/null || true
fi

if ! "${PYTHON_CMD[@]}" -c "import numpy, torch, omegaconf" >/dev/null 2>&1; then
  echo "ERROR: no compatible Python env found (requires numpy, torch, omegaconf)." >&2
  echo "DEBUG: which python -> $(which python)" >&2
  python -V >&2 || true
  exit 2
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_TAG=$(basename "${CHECKPOINT_DIR}")
OUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${PRESET}_${TASK}_${CKPT_TAG}"

CMD=(
  "${PYTHON_CMD[@]}" scripts/eval_continuous_capabilities.py
  --task "${TASK}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --checkpoint_pattern "**/checkpoints/best.ckpt"
  --preset "${PRESET}"
  --qa_file "configs/eval/continuous_qa_prompts.sample.jsonl"
  --fixed_texts_file "configs/eval/continuous_fixed_texts.jsonl"
  --output_dir "${OUT_DIR}"
  --save_samples
)

if [[ -n "${MAX_CHECKPOINTS}" ]]; then
  CMD+=(--max_checkpoints "${MAX_CHECKPOINTS}")
fi

echo "=========================================="
echo "Continuous Capability Evaluation - $(date)"
echo "Task: ${TASK}"
echo "Preset: ${PRESET}"
echo "Checkpoint root: ${CHECKPOINT_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "=========================================="

"${CMD[@]}"

echo "=========================================="
echo "Evaluation complete at $(date)"
echo "Results: ${OUT_DIR}/results.json"
echo "=========================================="
