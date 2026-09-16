# Retry wrapper for Jupiter ParaStation flaky first-spawn
# (PSI: doSpawn … Invalid argument / PSI_spawnRsrvtn).
#
# Usage: source scripts/_srun_retry.bash
#        srun_retry [srun args…]
#        srun_warmup              # cheap multi-node barrier before real work

srun_retry() {
  local max="${SRUN_RETRY_MAX:-8}"
  local delay="${SRUN_RETRY_DELAY:-2}"
  # Only retry failures that happen quickly (ParaStation spawn). App errors
  # mid-suite often surface as Force Terminated → rc≥128; retrying then
  # re-runs the entire lm-eval job and wipes hours of finished tasks.
  local window="${SRUN_RETRY_WINDOW_SEC:-120}"
  local n=1
  local rc=0
  local start=0
  local elapsed=0
  while true; do
    # Do NOT use `if srun; then` — a failed command inside `if` leaves $?=0
    # after the if-compound, which falsely reports success.
    start="$(date +%s)"
    set +e
    srun "$@"
    rc=$?
    set -e
    elapsed=$(( $(date +%s) - start ))
    if [[ "${rc}" -eq 0 ]]; then
      if [[ "${n}" -gt 1 ]]; then
        echo "[srun_retry] succeeded on attempt ${n}" >&2
      fi
      return 0
    fi
    echo "[srun_retry] attempt ${n}/${max} failed rc=${rc} after ${elapsed}s: srun $*" >&2
    # Application exit codes (<128) are real failures.
    if [[ "${rc}" -lt 128 ]]; then
      echo "[srun_retry] non-transient rc=${rc}; not retrying" >&2
      return "${rc}"
    fi
    # Slow SIGTERM/SIGHUP: treat as app-driven Force Terminated, not PSI.
    if [[ "${elapsed}" -ge "${window}" ]]; then
      echo "[srun_retry] failure after ${elapsed}s ≥ window ${window}s; not retrying (likely app error)" >&2
      return "${rc}"
    fi
    if [[ "${n}" -ge "${max}" ]]; then
      return "${rc}"
    fi
    sleep "${delay}"
    if [[ "${delay}" -lt 30 ]]; then
      delay=$((delay * 2))
      if [[ "${delay}" -gt 30 ]]; then delay=30; fi
    fi
    n=$((n + 1))
  done
}

srun_warmup() {
  # One task per node — establishes PSI spawn path before heavy launches.
  local nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
  if [[ "${nodes}" -le 1 ]]; then
    return 0
  fi
  echo "[srun_warmup] barrier across ${nodes} nodes…" >&2
  srun_retry --ntasks-per-node=1 --nodes="${nodes}" /usr/bin/hostname >/dev/null
}
