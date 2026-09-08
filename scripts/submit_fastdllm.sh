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
#   paper scale: seq 2048, block 32, GBS 256, lr 2e-5, **1900 steps ≈ 1B tokens**
#     (override MAX_STEPS=6000 for the longer ~3.15B stack cell)
#
# Remaining deltas (not "hard limits" — residual honesty):
#   Nemotron subset / packing may differ from paper's undisclosed slice
#   mask token string `[MASK]` vs official `|<MASK>|` (id still 151665)
#   fused CUDA / flex train kernels ≠ our SDPA (throughput only)
#   OOM fallback: EXTRA_OVERRIDES='algo.complementary_batching=sequential'
#
# Usage:
#   ./scripts/submit_fastdllm.sh           # exactness cell (~1B toks)
#   MAX_STEPS=6000 ./scripts/submit_fastdllm.sh
#   ./scripts/submit_fastdllm.sh --dry-run

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

# Paper claims ~1B tokens. Default to that budget for exactness cells:
#   1900 × GBS 256 × 2048 ≈ 0.997B tokens.
# Use MAX_STEPS=6000 for the longer stack-matched run (~3.15B).
export BLOCK="${BLOCK:-32}"
export SEQ_LEN="${SEQ_LEN:-2048}"
export GBS="${GBS:-256}"
export MAX_STEPS="${MAX_STEPS:-1900}"
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
