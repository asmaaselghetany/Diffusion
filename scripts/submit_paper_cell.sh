#!/usr/bin/env bash
# One entrypoint for paper design cells (PAPER_EXPERIMENTS.md).
#
# Usage:
#   ./scripts/submit_paper_cell.sh --list
#   ./scripts/submit_paper_cell.sh C0 --dry-run
#   ./scripts/submit_paper_cell.sh C2_fdllm
#   ./scripts/submit_paper_cell.sh B1_uniform
#   ./scripts/submit_paper_cell.sh B3_mixture --arm uniform
#   CKPT=outputs/.../last.ckpt ./scripts/submit_paper_cell.sh C4
#   CKPT=outputs/.../last.ckpt ./scripts/submit_paper_cell.sh lm_eval
#   ./scripts/submit_paper_cell.sh C5_joint_ar
#   ./scripts/submit_paper_cell.sh B4_hybrid_p10
#
# Train cells default to paper scale (6000/2048/256). Micros: --micro
# (forwarded to submit_lever.sh for lever cells only).

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi
if ! command -v python >/dev/null 2>&1; then
  echo "No python on PATH (expected ${REPO_ROOT}/.venv/bin/python)" >&2
  exit 1
fi

CELLS_YAML="${REPO_ROOT}/configs/paper/cells.yaml"
LIST=0
DRY_RUN=0
MICRO=0
ARM_OVERRIDE=""
CELL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list|-l) LIST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --micro) MICRO=1; shift ;;
    --arm) ARM_OVERRIDE="${2:?}"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    -*)
      echo "Unknown flag: $1" >&2
      exit 1
      ;;
    *)
      CELL="$1"
      shift
      ;;
  esac
done

if [[ "${LIST}" -eq 1 || -z "${CELL}" ]]; then
  python - "${CELLS_YAML}" <<'PY'
import sys, yaml
from pathlib import Path
reg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(f"{'CELL':<16} {'STATUS':<10} {'LAUNCH':<18} TITLE")
print("-" * 78)
for name, spec in (reg.get("cells") or {}).items():
  print(f"{name:<16} {spec.get('status','?'):<10} {str(spec.get('launch','-')):<18} {spec.get('title','')}")
print()
print("Ready train:   ./scripts/submit_paper_cell.sh <CELL>")
print("Eval helpers:  CKPT=... ./scripts/submit_paper_cell.sh C4|eval_ckpt|lm_eval|family_eval|E_*")
print("Family router: ./scripts/submit_family_eval.sh <ckpt>   # auto DualCache lm-eval vs ARPC")
print("Longitudinal:  RUN_ROOT=... ./scripts/submit_paper_cell.sh longitudinal")
PY
  [[ -n "${CELL}" ]] || exit 0
fi

# Resolve cell via Python (status, launch type, preset, arm).
META="$(python - "${CELLS_YAML}" "${CELL}" "${ARM_OVERRIDE}" <<'PY'
import json, sys, yaml
from pathlib import Path
reg = yaml.safe_load(Path(sys.argv[1]).read_text())
cell = sys.argv[2]
arm_ov = sys.argv[3] or None
cells = reg.get("cells") or {}
if cell not in cells:
  print(f"Unknown cell={cell!r}. Known: {sorted(cells)}", file=sys.stderr)
  sys.exit(2)
spec = dict(cells[cell])
spec["name"] = cell
if arm_ov:
  spec["arm"] = arm_ov
print(json.dumps(spec))
PY
)" || exit 2

STATUS="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "${META}")"
LAUNCH="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("launch",""))' "${META}")"
TITLE="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("title",""))' "${META}")"
NOTES="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("notes",""))' "${META}")"
REASON="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("reason",""))' "${META}")"
PRESET="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("preset") or "")' "${META}")"
ARM="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("arm") or "masked")' "${META}")"

echo "=== paper cell ${CELL} ==="
echo "  title:  ${TITLE}"
echo "  status: ${STATUS}"
echo "  launch: ${LAUNCH}"
[[ -n "${NOTES}" ]] && echo "  notes:  ${NOTES}"

if [[ "${STATUS}" == "stub" ]]; then
  echo "STUB: ${REASON:-not implemented}" >&2
  exit 3
fi

if [[ "${MICRO}" -eq 1 && "${LAUNCH}" != "lever" ]]; then
  echo "WARN: --micro is only forwarded for launch=lever cells; " \
       "${CELL} uses ${LAUNCH} (full sbatch scale)." >&2
fi

run_or_echo() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY_RUN: $*"
  else
    "$@"
  fi
}

case "${LAUNCH}" in
  ar2block_masked)
    run_or_echo sbatch scripts/slurm/ar2block_masked.sbatch
    ;;
  ar2block_uniform)
    run_or_echo sbatch scripts/slurm/ar2block_uniform.sbatch
    ;;
  block_masked)
    run_or_echo sbatch scripts/slurm/block_masked.sbatch
    ;;
  block_uniform)
    run_or_echo sbatch scripts/slurm/block_uniform.sbatch
    ;;
  ar_sft)
    # Matched causal AR SFT (paper cell A_ar_sft — Tab-1 AR reference).
    # Not full-seq diffusion; not C3.
    export NEMOTRON_SFT_SPLITS="${NEMOTRON_SFT_SPLITS:-chat,safety,science,math,code}"
    export NEMOTRON_SFT_MAX_PER_SPLIT="${NEMOTRON_SFT_MAX_PER_SPLIT:-math=1000000,code=500000}"
    export NUM_NODES="${NUM_NODES:-8}"
    export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
    export NUM_GPUS="${NUM_GPUS:-$((NUM_NODES * GPUS_PER_NODE))}"
    echo "  A_ar_sft scale: nodes=${NUM_NODES} gpus/node=${GPUS_PER_NODE} splits=${NEMOTRON_SFT_SPLITS} caps=${NEMOTRON_SFT_MAX_PER_SPLIT}"
    run_or_echo sbatch --nodes="${NUM_NODES}" --ntasks-per-node="${GPUS_PER_NODE}" \
      --gres=gpu:"${GPUS_PER_NODE}" --time=12:00:00 --export=ALL \
      scripts/slurm/ar_sft.sbatch
    ;;
  u0_from_c0)
    run_or_echo bash scripts/submit_u0_from_c0.sh
    ;;
  lever)
    if [[ -z "${PRESET}" ]]; then
      echo "Cell ${CELL} missing preset" >&2
      exit 1
    fi
    # Paper C3 full-seq diffusion: one block = model.length.
    if [[ "${CELL}" == "C3" || "${PRESET}" == "C3_fullseq" || "${PRESET}" == C3_fullseq_* ]]; then
      export BLOCK="${BLOCK:-2048}"
      echo "  C3 full-seq: BLOCK=${BLOCK} (must equal model.length)"
    fi
    args=(--preset "${PRESET}" --arm "${ARM}")
    if [[ "${MICRO}" -eq 1 ]]; then
      args+=(--micro)
    else
      args+=(--paper)
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      args+=(--dry-run)
    fi
    # Tab-2 / conversion levers: same math mix as canonical C0/C2-math.
    export NEMOTRON_SFT_SPLITS="${NEMOTRON_SFT_SPLITS:-chat,safety,science,math,code}"
    export NEMOTRON_SFT_MAX_PER_SPLIT="${NEMOTRON_SFT_MAX_PER_SPLIT:-math=1000000,code=500000}"
    ./scripts/submit_lever.sh "${args[@]}"
    ;;
  eval_checkpoint)
    if [[ -z "${CKPT:-}" || ! -f "${CKPT}" ]]; then
      echo "Set CKPT=/path/to.ckpt for eval_ckpt" >&2
      exit 1
    fi
    run_or_echo sbatch scripts/slurm/eval_checkpoint.sbatch "${CKPT}"
    ;;
  lm_eval)
    if [[ -z "${CKPT:-}" || ! -f "${CKPT}" ]]; then
      echo "Set CKPT=/path/to.ckpt for lm_eval" >&2
      exit 1
    fi
    # Prefer family router (auto DualCache vs baseline; refuses uniform).
    if [[ -x "${REPO_ROOT}/scripts/submit_family_eval.sh" ]]; then
      args=("${CKPT}" --lm-eval-only)
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        args+=(--dry-run)
      fi
      run_or_echo bash "${REPO_ROOT}/scripts/submit_family_eval.sh" "${args[@]}"
    else
      export CKPT
      export DECODE_PROFILE="${DECODE_PROFILE:-auto}"
      run_or_echo sbatch --export=ALL,CKPT,TASKS,NUM_STEPS,MAX_NEW_TOKENS,SKIP_THROUGHPUT,OUT_DIR,DECODE_PROFILE,UNMASK_THRESHOLD,FORCE_GREEDY,SUITE,NUM_NODES,GPUS_PER_NODE,LM_EVAL_DIST_TIMEOUT_SEC \
        scripts/slurm/lm_eval.sbatch
    fi
    ;;
  family_eval)
    if [[ -z "${CKPT:-}" || ! -f "${CKPT}" ]]; then
      echo "Set CKPT=/path/to.ckpt for family_eval" >&2
      exit 1
    fi
    args=("${CKPT}")
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      args+=(--dry-run)
    fi
    run_or_echo bash "${REPO_ROOT}/scripts/submit_family_eval.sh" "${args[@]}"
    ;;
  nfe_sweep)
    if [[ -z "${CKPT:-}" || ! -f "${CKPT}" ]]; then
      echo "Set CKPT=/path/to.ckpt for C4 NFE / GenPPL hygiene" >&2
      exit 1
    fi
    # Default C4 = full hygiene (dual + nfe + panel). Override MODE=nfe if needed.
    export MODE="${MODE:-all}"
    export NUM_STEPS_LIST="${NUM_STEPS_LIST:-8 16 32 64}"
    run_or_echo bash scripts/submit_gen_ppl_hygiene.sh "${CKPT}"
    ;;
  decode_eval)
    if [[ -z "${CKPT:-}" || ! -f "${CKPT}" ]]; then
      echo "Set CKPT=/path/to.ckpt for decode preset ${CELL}" >&2
      exit 1
    fi
    if [[ -z "${PRESET}" ]]; then
      echo "Cell ${CELL} missing preset" >&2
      exit 1
    fi
    run_or_echo python tools/run_decode_preset_eval.py \
      --checkpoint "${CKPT}" \
      --preset "${PRESET}" \
      --arm "${ARM}"
    ;;
  longitudinal_eval)
    if [[ -z "${RUN_ROOT:-}" || ! -d "${RUN_ROOT}" ]]; then
      echo "Set RUN_ROOT=/path/to/run_dir for longitudinal eval" >&2
      exit 1
    fi
    run_or_echo bash scripts/submit_longitudinal_eval.sh "${RUN_ROOT}"
    ;;
  *)
    echo "Unknown launch type: ${LAUNCH}" >&2
    exit 1
    ;;
esac
