#!/usr/bin/env bash
#SBATCH --job-name=qwen-base-ifeval-mmlu
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
# Base Qwen2.5-1.5B-Instruct IFEval + MMLU. Always 8x4 allocation.
# IFEval: 1 process (multi-GPU metric gather bug). MMLU: 32 processes.
#   sbatch scripts/slurm/qwen_base_ifeval_mmlu.sbatch

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
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
if [[ -d "${HF_HOME}/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots" ]]; then
  snap="$(ls -d "${HF_HOME}/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"* 2>/dev/null | head -1 || true)"
  if [[ -n "${snap}" && -f "${snap}/config.json" ]]; then
    MODEL="${snap}"
  fi
fi

TASKS="${TASKS:-ifeval,mmlu}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/qwen_base_instruct/ifeval_mmlu}"
mkdir -p "${OUT_DIR}" slurm_logs

NUM_NODES="${SLURM_JOB_NUM_NODES:-8}"
GPUS_PER_NODE=4
NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))
MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))

echo "=== qwen base IFEval/MMLU ==="
echo "  model=${MODEL} nodes=${NUM_NODES}x${GPUS_PER_NODE}"
echo "  tasks=${TASKS} out=${OUT_DIR}"

# shellcheck disable=SC1091
source scripts/_srun_retry.bash
srun_warmup

run_ifeval() {
  echo "=== task=ifeval nproc=1 (gather-safe) ===" | tee -a "${OUT_DIR}/lm_eval.log"
  # Single process on rank-0 node only; allocation stays 8x4 per site policy.
  srun_retry --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpu-bind=none \
    bash -c "
      source '${HUB_VENV}/bin/activate'
      export PATH='${HUB_VENV}/bin':\"\${PATH}\"
      export HF_HOME='${HF_HOME}' HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'
      export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
      CUDA_VISIBLE_DEVICES=0 python -u -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True \
        --tasks ifeval \
        --batch_size 8 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path '${OUT_DIR}/ifeval'
    " 2>&1 | tee -a "${OUT_DIR}/lm_eval.log"
}

run_mmlu() {
  echo "=== task=mmlu nproc=${NUM_PROCESSES} ===" | tee -a "${OUT_DIR}/lm_eval.log"
  srun_retry --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
    --cpu-bind=none \
    bash -c "
      source '${HUB_VENV}/bin/activate'
      export PATH='${HUB_VENV}/bin':\"\${PATH}\"
      export HF_HOME='${HF_HOME}' HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'
      export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
      export NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0
      accelerate launch \
        --num_processes ${NUM_PROCESSES} \
        --num_machines ${NUM_NODES} \
        --machine_rank \"\${SLURM_NODEID:-0}\" \
        --main_process_ip '${MASTER_ADDR}' \
        --main_process_port ${MASTER_PORT} \
        --mixed_precision no \
        -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True \
        --tasks mmlu \
        --batch_size 1 \
        --num_fewshot 5 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path '${OUT_DIR}/mmlu'
    " 2>&1 | tee -a "${OUT_DIR}/lm_eval.log"
}

IFS=',' read -r -a TASK_ARR <<< "${TASKS}"
for t in "${TASK_ARR[@]}"; do
  t="$(echo "${t}" | xargs)"
  [[ -n "${t}" ]] || continue
  case "${t}" in
    ifeval) run_ifeval ;;
    mmlu) run_mmlu ;;
    *) echo "unsupported task ${t}" >&2; exit 1 ;;
  esac
done

echo "Done. Results under ${OUT_DIR}"
