#!/usr/bin/env bash
# C4: NFE × max_new_tokens sweep on one conversion checkpoint.
#
# Usage:
#   CKPT=outputs/.../last.ckpt ./scripts/submit_nfe_sweep.sh
#   CKPT=... NUM_STEPS_LIST="8 16 32 64" MAX_NEW_TOKENS_LIST="128 512" \
#     ./scripts/submit_nfe_sweep.sh
#
# Writes under ``<run_root>/eval/nfe_sweep/steps_<N>_mnt_<M>/``.

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

CKPT="${CKPT:-${1:-}}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Usage: CKPT=/path/to.ckpt $0" >&2
  exit 1
fi
CKPT="$(readlink -f "${CKPT}")"

NUM_STEPS_LIST="${NUM_STEPS_LIST:-8 16 32 64}"
MAX_NEW_TOKENS_LIST="${MAX_NEW_TOKENS_LIST:-512}"
RUN_ROOT="$(dirname "$(dirname "${CKPT}")")"
OUT_ROOT="${NFE_OUT_DIR:-${RUN_ROOT}/eval/nfe_sweep}"
mkdir -p "${OUT_ROOT}"

# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

echo "=== NFE sweep ==="
echo "  ckpt:  ${CKPT}"
echo "  out:   ${OUT_ROOT}"
echo "  steps: ${NUM_STEPS_LIST}"
echo "  mnt:   ${MAX_NEW_TOKENS_LIST}"

for steps in ${NUM_STEPS_LIST}; do
  for mnt in ${MAX_NEW_TOKENS_LIST}; do
    sub="${OUT_ROOT}/steps_${steps}_mnt_${mnt}"
    mkdir -p "${sub}"
    echo "--- steps=${steps} max_new_tokens=${mnt} -> ${sub} ---"
    NUM_STEPS="${steps}" \
    MAX_NEW_TOKENS="${mnt}" \
    NUM_SAMPLES="${NUM_SAMPLES:-32}" \
    BATCH_SIZE="${BATCH_SIZE:-1}" \
    bash examples/block_qwen/eval.sh "${CKPT}" "${sub}"
    OUT_DIR="${sub}/lm_eval" \
    mkdir -p "${OUT_DIR}" \
    TASKS="${TASKS:-gsm8k,ifeval}" \
    NUM_STEPS="${steps}" \
    MAX_NEW_TOKENS="${mnt}" \
    SKIP_THROUGHPUT="${SKIP_THROUGHPUT:-0}" \
    bash examples/block_qwen/lm_eval.sh "${CKPT}"
    python - "${sub}" <<'PY'
import json, sys
from pathlib import Path
sub = Path(sys.argv[1])
manifest = {
    "num_steps": int(sub.name.split("_mnt_")[0].split("_")[-1]),
    "max_new_tokens": int(sub.name.split("_mnt_")[-1]),
    "gen_ppl": str(sub / "gen_ppl_metrics.json") if (sub / "gen_ppl_metrics.json").is_file() else None,
    "lm_eval_summary": str(sub / "lm_eval" / "SUMMARY.json") if (sub / "lm_eval" / "SUMMARY.json").is_file() else None,
}
(sub / "nfe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
  done
done

echo "Done. Sweep results under ${OUT_ROOT}"
