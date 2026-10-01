#!/usr/bin/env bash
#SBATCH --job-name=qwen-base-mmlu
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Base Qwen2.5-1.5B-Instruct MMLU only (IFEval already done @ ~39.4%).
# 8-node allocation (site policy); eval is 1-process to avoid lm-eval
# metric_list=None gather crash under Accelerate+hf.
#   sbatch scripts/slurm/qwen_base_mmlu_only.sbatch

set -euo pipefail

export ASMAA_WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${ASMAA_WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/jupiter_paths.sh"
activate_jupiter_modules

HUB_VENV="${HUB_VENV:-/e/scratch/scifi/elsayed3/.venvs/hub_fastdllm}"
# shellcheck disable=SC1091
source "${HUB_VENV}/bin/activate"
export PATH="${HUB_VENV}/bin:${PATH}"

export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
if [[ -d "${HF_HOME}/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots" ]]; then
  snap="$(ls -d "${HF_HOME}/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"* 2>/dev/null | head -1 || true)"
  if [[ -n "${snap}" && -f "${snap}/config.json" ]]; then
    MODEL="${snap}"
  fi
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/qwen_base_instruct/ifeval_mmlu}"
mkdir -p "${OUT_DIR}" slurm_logs

echo "=== qwen base MMLU-only ==="
echo "  model=${MODEL}"
echo "  out=${OUT_DIR} (8-node alloc; 1-proc eval)"

# shellcheck disable=SC1091
source scripts/_srun_retry.bash
srun_warmup

# Rank-0 node, single GPU — gather-safe.
srun_retry --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpu-bind=none \
  bash -c "
    source '${HUB_VENV}/bin/activate'
    export PATH='${HUB_VENV}/bin':\"\${PATH}\"
    export HF_HOME='${HF_HOME}' HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    CUDA_VISIBLE_DEVICES=0 python -u -m lm_eval \
      --model hf \
      --model_args pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True \
      --tasks mmlu \
      --batch_size 1 \
      --num_fewshot 5 \
      --apply_chat_template \
      --fewshot_as_multiturn \
      --output_path '${OUT_DIR}/mmlu'
  " 2>&1 | tee -a "${OUT_DIR}/lm_eval.log"

echo "Done. Results under ${OUT_DIR}/mmlu"
