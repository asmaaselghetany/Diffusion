#!/usr/bin/env bash
# Helper for qwen_base_ifeval_mmlu.sbatch (runs under srun on each node).
set -euo pipefail
# shellcheck disable=SC1091
source "${HUB_VENV}/bin/activate"
export PATH="${HUB_VENV}/bin:${PATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"

fewshot_args=()
if [[ -n "${FEWSHOT_N:-}" ]]; then
  # FEWSHOT_N like "--num_fewshot 5"
  # shellcheck disable=SC2206
  fewshot_args=(${FEWSHOT_N})
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_NODES}" \
  --machine_rank "${SLURM_NODEID:-0}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision no \
  -m lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True" \
  --tasks "${TASK}" \
  --batch_size "${BATCH}" \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path "${OUT_DIR}/${TASK}" \
  "${fewshot_args[@]}"
