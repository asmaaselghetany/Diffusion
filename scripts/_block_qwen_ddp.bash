# Shared Slurm/Lightning DDP resource wiring for block_qwen jobs.
# Source from _block_qwen_launch.bash and ar_sft.sbatch.
#
# Codex pattern: one Slurm task per GPU (external process group). Never combine
# trainer.devices=N with --gpus-per-task=1 on the same node.

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

append_block_qwen_trainer_overrides() {
  # Lightning: devices = GPUs per node; num_nodes = Slurm nodes.
  _append_override "trainer.devices" "${GPUS_PER_NODE}"
  _append_override "trainer.num_nodes" "${NUM_NODES}"
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    _append_override "strategy" "ddp"
  fi
  _append_override "loader.eval_global_batch_size" "${NUM_GPUS}"
}

run_block_qwen_srun_train() {
  local train_rc=0
  if [[ "${NUM_NODES}" -gt 1 ]]; then
    # Multi-node: one Slurm task per GPU; Lightning uses the external process group.
    srun --ntasks-per-node="${GPUS_PER_NODE}" --cpu-bind=cores \
      python -u -m discrete_diffusion "+experiment=${EXPERIMENT}" "algo=${ALGO}" \
      data.cache_dir="${DATA_CACHE}" \
      checkpointing.save_dir="${RUN_ROOT}" \
      checkpointing.resume_from_ckpt="${RESUME_FROM_CKPT:-true}" \
      hydra.run.dir="${RUN_ROOT}/hydra" \
      "wandb.project=${WANDB_PROJECT}" \
      "wandb.name=${WANDB_RUN_NAME}" \
      "wandb.id=${WANDB_RUN_ID}" \
      "wandb.resume=${WANDB_RESUME}" \
      "${EXTRA_OVERRIDES[@]}" || train_rc=$?
  else
    # Single-node: still launch one task per GPU. SlurmEnvironment does not
    # spawn sibling ranks from a lone task (ntasks=1 + devices=2 wastes GPUs).
    # Do not add --gpus-per-task=1: Lightning validates devices against all
    # visible GPUs, then binds each rank via LOCAL_RANK.
    srun --ntasks="${GPUS_PER_NODE}" --ntasks-per-node="${GPUS_PER_NODE}" \
      --cpu-bind=cores \
      python -u -m discrete_diffusion "+experiment=${EXPERIMENT}" "algo=${ALGO}" \
      data.cache_dir="${DATA_CACHE}" \
      checkpointing.save_dir="${RUN_ROOT}" \
      checkpointing.resume_from_ckpt="${RESUME_FROM_CKPT:-true}" \
      hydra.run.dir="${RUN_ROOT}/hydra" \
      "wandb.project=${WANDB_PROJECT}" \
      "wandb.name=${WANDB_RUN_NAME}" \
      "wandb.id=${WANDB_RUN_ID}" \
      "wandb.resume=${WANDB_RESUME}" \
      "${EXTRA_OVERRIDES[@]}" || train_rc=$?
  fi
  return "${train_rc}"
}
