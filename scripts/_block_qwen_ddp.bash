# Shared Slurm/Lightning DDP resource wiring for block_qwen jobs.
# Source from _block_qwen_launch.bash and ar_sft.sbatch.
#
# Proven pattern (job 1701099): one Slurm task per GPU, trainer.devices =
# GPUs/node, all node GPUs visible, LOCAL_RANK selects the device.
# Do NOT use --gpus-per-task=1 with devices=N (exclusive CVD=[0] breaks
# parallel_devices[local_rank] and world-size = nodes*devices).

_BLOCK_QWEN_DDP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_block_qwen_ddp_resources() {
  NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
  if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
      GPUS_PER_NODE="${SLURM_GPUS_ON_NODE}"
    elif [[ -n "${NUM_GPUS:-}" && "${NUM_NODES}" -le 1 ]]; then
      # Backward compat: sbatch files that only export NUM_GPUS=2.
      GPUS_PER_NODE="${NUM_GPUS}"
    else
      GPUS_PER_NODE=2
    fi
  fi
  NUM_GPUS="${NUM_GPUS:-$((NUM_NODES * GPUS_PER_NODE))}"
  export NUM_NODES GPUS_PER_NODE NUM_GPUS
}

_force_override() {
  # Replace key=… if present; else append.
  local key="$1"
  local value="$2"
  if [[ "${HYDRA_OVERRIDES:-}" == *"${key}="* ]]; then
    HYDRA_OVERRIDES="$(python - "${HYDRA_OVERRIDES}" "${key}" "${value}" <<'PY'
import shlex, sys
old, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
out = []
for tok in shlex.split(old):
  if tok.startswith(key + "="):
    out.append(f"{key}={value}")
  else:
    out.append(tok)
print(" ".join(shlex.quote(t) for t in out))
PY
)"
  else
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES:+${HYDRA_OVERRIDES} }${key}=${value}"
  fi
}

append_block_qwen_trainer_overrides() {
  # Lightning: devices = GPUs per node; num_nodes = Slurm nodes.
  # World size = devices * num_nodes (= SLURM_NTASKS with one task/GPU).
  _force_override "trainer.devices" "${GPUS_PER_NODE}"
  _force_override "trainer.num_nodes" "${NUM_NODES}"
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    _force_override "strategy" "ddp"
  fi
  _force_override "loader.eval_global_batch_size" "${NUM_GPUS}"
}

# Slurm prologs often set CUDA_VISIBLE_DEVICES to a *single* GPU per task.
# Assigning CVD=0,1,2,3 *without unsetting first* only remaps within that
# singleton → Lightning sees 1 device and dies with
#   "You requested gpu: [0,1,2,3] But your machine only has: [0]"
# (C3 job 1856100). Unset, then expose all node GPUs; LOCAL_RANK picks one.
_cuda_visible_all_node_gpus_value() {
  local last=$((GPUS_PER_NODE - 1))
  seq -s, 0 "${last}"
}

_cuda_visible_all_node_gpus() {
  echo "CUDA_VISIBLE_DEVICES=$(_cuda_visible_all_node_gpus_value)"
}

run_block_qwen_srun_train() {
  local train_rc=0
  local py_bin
  local cvd_val
  py_bin="$(command -v python)"
  # Absolute path — ParaStation spawn is less brittle than PATH lookup.
  if [[ "${py_bin}" != /* ]]; then
    py_bin="$(readlink -f "${py_bin}" 2>/dev/null || true)"
  fi
  py_bin="${py_bin:-python}"
  cvd_val="$(_cuda_visible_all_node_gpus_value)"

  # Jupiter ParaStation: first multi-node srun often fails with
  #   PSI: doSpawn … Invalid argument / PSI_spawnRsrvtn
  # Warm up + retry (see scripts/_srun_retry.bash).
  # shellcheck disable=SC1091
  source "${_BLOCK_QWEN_DDP_DIR}/_srun_retry.bash"
  srun_warmup

  # --gpu-bind=none: all tasks on a node may see all allocated GPUs (needed
  # for trainer.devices=GPUS_PER_NODE with one srun task per GPU).
  if [[ "${NUM_NODES}" -gt 1 ]]; then
    # Multi-node: one Slurm task per GPU; Lightning uses the external process group.
    srun_retry --ntasks-per-node="${GPUS_PER_NODE}" --cpu-bind=cores \
      --gpu-bind=none \
      env -u CUDA_VISIBLE_DEVICES -u SLURM_CUDA_VISIBLE_DEVICES \
      "CUDA_VISIBLE_DEVICES=${cvd_val}" \
      "${py_bin}" -u -m discrete_diffusion "+experiment=${EXPERIMENT}" "algo=${ALGO}" \
      data.cache_dir="${DATA_CACHE}" \
      checkpointing.save_dir="${RUN_ROOT}" \
      checkpointing.resume_from_ckpt="${RESUME_FROM_CKPT:-true}" \
      hydra.run.dir="${RUN_ROOT}/hydra" \
      "wandb.project=${WANDB_PROJECT}" \
      "wandb.entity=${WANDB_ENTITY:-aselghetany-nu}" \
      "wandb.name=${WANDB_RUN_NAME}" \
      "wandb.id=${WANDB_RUN_ID}" \
      "wandb.resume=${WANDB_RESUME}" \
      "${EXTRA_OVERRIDES[@]}" || train_rc=$?
  else
    # Single-node: still launch one task per GPU.
    srun_retry --ntasks="${GPUS_PER_NODE}" --ntasks-per-node="${GPUS_PER_NODE}" \
      --cpu-bind=cores \
      --gpu-bind=none \
      env -u CUDA_VISIBLE_DEVICES -u SLURM_CUDA_VISIBLE_DEVICES \
      "CUDA_VISIBLE_DEVICES=${cvd_val}" \
      "${py_bin}" -u -m discrete_diffusion "+experiment=${EXPERIMENT}" "algo=${ALGO}" \
      data.cache_dir="${DATA_CACHE}" \
      checkpointing.save_dir="${RUN_ROOT}" \
      checkpointing.resume_from_ckpt="${RESUME_FROM_CKPT:-true}" \
      hydra.run.dir="${RUN_ROOT}/hydra" \
      "wandb.project=${WANDB_PROJECT}" \
      "wandb.entity=${WANDB_ENTITY:-aselghetany-nu}" \
      "wandb.name=${WANDB_RUN_NAME}" \
      "wandb.id=${WANDB_RUN_ID}" \
      "wandb.resume=${WANDB_RESUME}" \
      "${EXTRA_OVERRIDES[@]}" || train_rc=$?
  fi
  return "${train_rc}"
}
