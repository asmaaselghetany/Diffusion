#!/usr/bin/env bash
# Unifusion-style GenPPL hygiene eval (high value / low cost).
#
# Always reports GenPPL with unigram entropy (never PPL alone).
#
# Modes (MODE=...):
#   dual       — score existing samples.pt as first_chunk_only + full continuation
#   nfe        — regenerate at NUM_STEPS_LIST={8,16,32,64}, score both modes at 32 only? 
#                default: first_chunk_only=true per step (pair+H); optional DUAL_AT_DEFAULT=1
#   multiseed  — SEEDS="1 2 3" × NUM_SAMPLES (default 64) at fixed NUM_STEPS
#   panel      — collapse panel only (CPU-ish; still needs metrics+samples.txt)
#   all        — dual + nfe + panel on one ckpt (multiseed opt-in via RUN_MULTISEED=1)
#
# Usage:
#   CKPT=... MODE=dual  bash scripts/run_gen_ppl_hygiene.sh
#   CKPT=... MODE=nfe   bash scripts/run_gen_ppl_hygiene.sh
#   CKPT=... MODE=all   bash scripts/run_gen_ppl_hygiene.sh
#   CKPT=... MODE=multiseed SEEDS="1 2 3" bash scripts/run_gen_ppl_hygiene.sh
#
# Optional base-LM slice (bandwidth):
#   RUN_WINO=1 TASKS=winogrande bash …  (submitted separately via lm_eval; see notes)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
# Prefer shared env when under Slurm; allow login-node dry paths.
if [[ -f scripts/_block_qwen_env.bash && -n "${SLURM_JOB_ID:-}" ]]; then
  # shellcheck disable=SC1091
  source scripts/_block_qwen_env.bash
else
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
fi

CKPT="${CKPT:-${1:-}}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Usage: CKPT=/path/to.ckpt MODE=dual|nfe|multiseed|panel|all $0" >&2
  exit 1
fi
CKPT="$(readlink -f "${CKPT}")"
MODE="${MODE:-all}"
RUN_ROOT="$(dirname "$(dirname "${CKPT}")")"
OUT_ROOT="${HYGIENE_OUT:-${RUN_ROOT}/eval/unifusion_hygiene}"
mkdir -p "${OUT_ROOT}"

NUM_STEPS_LIST="${NUM_STEPS_LIST:-8 16 32 64}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_STEPS_DEFAULT="${NUM_STEPS:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SAMPLE_MODE="${SAMPLE_MODE:-conversion_free}"
# Default: BlockGen block skeleton. Lever callers pass hubmatch/dual_cache/UCC.
DECODE_PROFILE="${DECODE_PROFILE:-hierarchical}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEEDS="${SEEDS:-1 2 3}"
DUAL_AT_DEFAULT="${DUAL_AT_DEFAULT:-1}"
if [[ "${DECODE_PROFILE}" == "baseline" ]]; then
  DECODE_PROFILE=hierarchical
  echo "NOTE: baseline → hierarchical for GenPPL hygiene." >&2
fi
echo "  decode_profile=${DECODE_PROFILE}"

_samples_match_profile() {
  local samples_pt="$1"
  local want="${DECODE_PROFILE}"
  local meta="${samples_pt%.pt}.meta.json"
  if [[ ! -f "${samples_pt}" || ! -f "${meta}" ]]; then
    return 1
  fi
  python - "${meta}" "${want}" <<'PY'
import json, sys
meta = json.loads(open(sys.argv[1]).read())
want = sys.argv[2]
raise SystemExit(0 if str(meta.get("decode_profile")) == want else 1)
PY
}

_score_pair() {
  local samples_pt="$1"
  local metrics_json="$2"
  local first_chunk="$3"  # true|false
  python -u -m discrete_diffusion.evaluations.generative_ppl \
    --config-name=gen_ppl_block_qwen \
    "samples_path=${samples_pt}" \
    "metrics_path=${metrics_json}" \
    pretrained_model=gpt2-large \
    retokenize=true \
    "first_chunk_only=${first_chunk}"
  python tools/gen_ppl_hygiene.py pair "${metrics_json}"
}

_gen_samples() {
  local out_dir="$1"
  local steps="$2"
  local seed="${3:-}"
  mkdir -p "${out_dir}"
  local seed_args=()
  if [[ -n "${seed}" ]]; then
    seed_args+=("seed=${seed}")
  fi
  FORCE_REGEN=1 NUM_STEPS="${steps}" NUM_SAMPLES="${NUM_SAMPLES}" \
  SAMPLE_MODE="${SAMPLE_MODE}" DECODE_PROFILE="${DECODE_PROFILE}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" BATCH_SIZE="${BATCH_SIZE}" \
  bash examples/block_qwen/eval.sh "${CKPT}" "${out_dir}"
  # eval.sh already scores first_chunk_only=true; ensure pair sidecar
  python tools/gen_ppl_hygiene.py pair "${out_dir}/gen_ppl_metrics.json" \
    || true
}

echo "=== GenPPL hygiene (${MODE}) ==="
echo "  ckpt: ${CKPT}"
echo "  out:  ${OUT_ROOT}"

# Validate MODE *before* the fallthrough case. Patterns use ``;;&`` so a
# successful ``nfe|dual|…`` match continues testing later arms; a trailing
# ``*)`` would then fire and exit 1 after the real work finished
# (2083688: NFE 8/16/32/64 done, then ``Unknown MODE=nfe``).
case "${MODE}" in
  dual|nfe|multiseed|panel|all) ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 1
    ;;
esac

case "${MODE}" in
  dual|all)
    BASE_EVAL="${RUN_ROOT}/eval"
    SAMPLES="${BASE_EVAL}/samples.pt"
    # Prefer hygiene-local samples when base eval was generated under a
    # different decode_profile (old U0 hygiene used illegal baseline).
    if ! _samples_match_profile "${SAMPLES}"; then
      echo "Base eval samples missing or decode_profile≠${DECODE_PROFILE}; regenerating under hygiene out"
      SAMPLES_DIR="${OUT_ROOT}/samples_${DECODE_PROFILE}"
      _gen_samples "${SAMPLES_DIR}" "${NUM_STEPS_DEFAULT}"
      SAMPLES="${SAMPLES_DIR}/samples.pt"
    fi
    DUAL_DIR="${OUT_ROOT}/dual"
    mkdir -p "${DUAL_DIR}"
    echo "--- dual: first_chunk_only=true ---"
    _score_pair "${SAMPLES}" "${DUAL_DIR}/gen_ppl_metrics_first_chunk.json" true
    echo "--- dual: first_chunk_only=false (full continuation) ---"
    _score_pair "${SAMPLES}" "${DUAL_DIR}/gen_ppl_metrics_full.json" false
    python - "${DUAL_DIR}" "${DECODE_PROFILE}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
profile = sys.argv[2]
def load(name):
  d = json.loads((root / name).read_text())
  return {
    "ppl": d.get("ppl"),
    "H_mean": d.get("unigram_entropy_mean"),
    "H_med": d.get("unigram_entropy_median"),
    "first_chunk_only": d.get("first_chunk_only"),
    "honesty_warning": d.get("honesty_warning"),
  }
summary = {
  "protocol": "dual GenPPL modes on identical samples.pt",
  "decode_profile": profile,
  "first_chunk_only": load("gen_ppl_metrics_first_chunk.json"),
  "full_continuation": load("gen_ppl_metrics_full.json"),
  "note": "Do not mix modes in one table column. Cite (PPL, H) never PPL alone.",
}
(root / "dual_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("first_chunk:", summary["first_chunk_only"])
print("full:       ", summary["full_continuation"])
PY
    if [[ "${MODE}" == "dual" ]]; then
      :
    fi
    ;;&
  nfe|all)
    NFE_DIR="${OUT_ROOT}/nfe_sweep"
    mkdir -p "${NFE_DIR}"
    for steps in ${NUM_STEPS_LIST}; do
      sub="${NFE_DIR}/steps_${steps}"
      echo "--- NFE steps=${steps} -> ${sub} ---"
      _gen_samples "${sub}" "${steps}"
      if [[ "${DUAL_AT_DEFAULT}" == "1" && "${steps}" == "${NUM_STEPS_DEFAULT}" ]]; then
        _score_pair "${sub}/samples.pt" "${sub}/gen_ppl_metrics_full.json" false
      fi
    done
    python tools/gen_ppl_hygiene.py aggregate-nfe "${NFE_DIR}"
    ;;&
  multiseed|all)
    if [[ "${MODE}" == "all" && "${RUN_MULTISEED:-0}" != "1" ]]; then
      echo "(skip multiseed: set RUN_MULTISEED=1 to enable in MODE=all)"
    else
      MS_DIR="${OUT_ROOT}/multiseed"
      mkdir -p "${MS_DIR}"
      for seed in ${SEEDS}; do
        sub="${MS_DIR}/seed_${seed}"
        echo "--- multiseed seed=${seed} -> ${sub} ---"
        # generate_samples may accept seed via hydra; also set PYTHONHASHSEED
        export PYTHONHASHSEED="${seed}"
        mkdir -p "${sub}"
        python -u -m discrete_diffusion.evaluations.generate_samples \
          "checkpoint_path=${CKPT}" \
          "samples_path=${sub}/samples.pt" \
          "num_samples=${NUM_SAMPLES}" \
          "batch_size=${BATCH_SIZE}" \
          "num_steps=${NUM_STEPS_DEFAULT}" \
          "sample_mode=${SAMPLE_MODE}" \
          "decode_profile=${DECODE_PROFILE}" \
          "max_new_tokens=${MAX_NEW_TOKENS}" \
          "seed=${seed}" \
          save_text=true \
          device=cuda
        _score_pair "${sub}/samples.pt" "${sub}/gen_ppl_metrics.json" true
      done
      python tools/gen_ppl_hygiene.py aggregate-multiseed "${MS_DIR}"
    fi
    ;;&
  panel|all)
    PANEL_DIR="${OUT_ROOT}/collapse_panel"
    mkdir -p "${PANEL_DIR}"
    # Prefer default eval samples; else NFE steps_32
    SRC_TXT="${RUN_ROOT}/eval/samples.txt"
    SRC_MET="${RUN_ROOT}/eval/gen_ppl_metrics.json"
    if [[ ! -f "${SRC_TXT}" || ! -f "${SRC_MET}" ]]; then
      SRC_TXT="${OUT_ROOT}/nfe_sweep/steps_${NUM_STEPS_DEFAULT}/samples.txt"
      SRC_MET="${OUT_ROOT}/nfe_sweep/steps_${NUM_STEPS_DEFAULT}/gen_ppl_metrics.json"
    fi
    if [[ -f "${SRC_TXT}" && -f "${SRC_MET}" ]]; then
      # Ensure metrics have entropy
      python tools/gen_ppl_hygiene.py pair "${SRC_MET}" || \
        _score_pair "$(dirname "${SRC_MET}")/samples.pt" "${SRC_MET}" true
      python tools/gen_ppl_hygiene.py collapse-panel \
        --samples-txt "${SRC_TXT}" \
        --metrics "${SRC_MET}" \
        --out "${PANEL_DIR}/panel.json"
    else
      echo "WARNING: no samples.txt/metrics for collapse panel" >&2
    fi
    ;;
esac

# Top-level index
python - "${OUT_ROOT}" "${CKPT}" "${MODE}" "${DECODE_PROFILE}" <<'PY'
import json, sys
from pathlib import Path
root, ckpt, mode, profile = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
idx = {
  "checkpoint": ckpt,
  "mode": mode,
  "decode_profile": profile,
  "artifacts": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()),
  "rule": "GenPPL must appear as (ppl, H_mean[, H_med]) — never alone",
}
(root / "hygiene_index.json").write_text(json.dumps(idx, indent=2) + "\n")
print(f"Wrote {root}/hygiene_index.json ({len(idx['artifacts'])} files)")
PY

echo "Done. Hygiene artifacts under ${OUT_ROOT}"
