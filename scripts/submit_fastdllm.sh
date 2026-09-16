# Fast-dLLM v2 — one-to-one recipe on our Qwen conversion stack.
#
# Official refs:
#   https://github.com/NVlabs/Fast-dLLM (v2/)  arXiv:2509.26328
#   third_party/Fast-dLLM + Hub modeling.py (v2/hub_ref/)
#
# Paper / public recipe (1.5B):
#   Qwen2.5-Instruct → block diffusion SFT on LLaMA-Nemotron (~1B tokens)
#   block size 32, shifted CE, complementary m/~m (fused 2B), unweighted CE
#   decode: single-stream block-causal + hierarchical / DualCache
#
# Our wiring (scripts/submit_fastdllm.sh → preset fastdllm_v2):
#   line=ar2block (AR pretrained init), algo=block_masked
#   data=sft_qwen (Nemotron — same family as paper §4.1)
#   levers: shift + complementary(fused) + mask_schedule=fast_dllm + plain_ce
#           + hierarchical_kv + dual_cache + single_stream_decode
#           + fdllm_confidence_decode (thr=0.9, greedy)
#   paper scale (appendix): seq 2048, block 32, GBS 256, lr 2e-5,
#     **6000 steps ≈ 3.15B tokens** (1.5B recipe that produced Table 1).
#     Abstract "~1B" is marketing vs Dream; use MAX_STEPS=1900 only for short probes.
#
# Remaining irreducible deltas:
#   Nemotron subset / packing may differ from paper's undisclosed slice
#   mask token string `[MASK]` vs official `|<MASK>|` (id still 151665)
#   fused CUDA / flex train kernels ≠ our SDPA (throughput only)
#   OOM fallback: EXTRA_OVERRIDES='algo.complementary_batching=sequential'
#
# Usage:
#   ./scripts/submit_fastdllm.sh           # paper-schedule 6k (~3.15B)
#   MAX_STEPS=1900 ./scripts/submit_fastdllm.sh   # short ~1B probe
#   ./scripts/submit_fastdllm.sh --dry-run
#
# Enhanced conversion baseline (not levers) — now default in
# discrete_diffusion.data.conversion_baseline:
#   Nemotron chat+safety+science+math+code (math≤1M, code≤500k; v3.1-tplhash)
#   Hub-like vocab keep 151936; train↔eval ChatML; shared fingerprint (+ template hash)
# Override splits: NEMOTRON_SFT_SPLITS=chat,safety,science

set -euo pipefail
WORKSPACE="${ASMAA_WORKSPACE:-/e/project1/scifi/elsayed3}"
# shellcheck disable=SC1091
source "${WORKSPACE}/env.sh"
cd "${REPO_ROOT}"

DRY_ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_ARGS+=(--dry-run) ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
  esac
done

# Nemotron SFT cache (do not inherit BlockGen OWT DATA_CACHE).
export DATA_CACHE="${FASTDLLM_DATA_CACHE:-${DISCRETE_DIFFUSION_SCRATCH_DIR}/block_qwen_sft_nemotron}"
mkdir -p "${DATA_CACHE}"

# Paper appendix 1.5B schedule (Table 1): 6000 × 256 × 2048 ≈ 3.15B tokens.
# Short probe: MAX_STEPS=1900 (~1B, abstract slogan only).
export BLOCK="${BLOCK:-32}"
export SEQ_LEN="${SEQ_LEN:-2048}"
export GBS="${GBS:-256}"
export MAX_STEPS="${MAX_STEPS:-6000}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export NUM_NODES="${NUM_NODES:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export WANDB_PROJECT="${WANDB_PROJECT:-block_qwen}"

# Optional non-lever extras (chat template / tokenizer already set by experiment).
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  export EXTRA_OVERRIDES="${EXTRA_OVERRIDES}"
fi

exec ./scripts/submit_lever.sh \
  --preset fastdllm_v2 \
  --arm masked \
  --line ar2block \
  --paper \
  "${DRY_ARGS[@]}"
