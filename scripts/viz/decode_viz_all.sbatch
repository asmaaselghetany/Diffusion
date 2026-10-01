#!/usr/bin/env bash
#SBATCH --job-name=decode_viz_all
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Record + render decode GIFs for all scoreboard-relevant BlockTrainer ckpts.

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

OUT_ROOT="${OUT_ROOT:-docs/research/decode_viz}"
PROMPT="${PROMPT:-Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?}"
MAX_NEW="${MAX_NEW_TOKENS:-96}"
STEPS="${NUM_STEPS:-32}"
SEED="${SEED:-0}"

run_one() {
  local name="$1" ckpt="$2" profile="$3" thr="${4:-}"
  local args=(
    --name "${name}"
    --ckpt "${ckpt}"
    --profile "${profile}"
    --prompt "${PROMPT}"
    --device cuda
    --max-new "${MAX_NEW}"
    --steps "${STEPS}"
  )
  if [[ -n "${thr}" ]]; then
    args+=(--thr "${thr}")
  fi
  echo "=== ${name} (${profile}${thr:+ thr=${thr}}) ==="
  if [[ ! -f "${ckpt}" ]]; then
    echo "SKIP missing ckpt: ${ckpt}"
    return 0
  fi
  # record_decode_trace defaults to --no-greedy (good for uniform);
  # hubmatch profile sets greedy=true via hydra overrides inside load.
  ./scripts/viz/make_decode_demo.sh "${args[@]}"
}

# ---- Masked / hubmatch thr=1 ----
run_one baseline_masked \
  outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt \
  hubmatch 1.0

run_one shift_masked \
  outputs/block_qwen/ar2block_masked_1856101/checkpoints/last.ckpt \
  hubmatch 1.0

run_one complementary_masked \
  outputs/block_qwen/ar2block_masked_1856103/checkpoints/last.ckpt \
  hubmatch 1.0

run_one shift_comp_masked \
  outputs/block_qwen/ar2block_masked_1856105/checkpoints/last.ckpt \
  hubmatch 1.0

run_one anneal_masked \
  outputs/block_qwen/ar2block_masked_1857473/checkpoints/last.ckpt \
  hubmatch 1.0

run_one joint_masked \
  outputs/block_qwen/ar2block_masked_1836723/checkpoints/last.ckpt \
  hubmatch 1.0

run_one allfive_joint_masked \
  outputs/block_qwen/ar2block_masked_1836725/checkpoints/last.ckpt \
  hubmatch 1.0

run_one mathmix_masked \
  outputs/block_qwen/ar2block_masked_1763301/checkpoints/last.ckpt \
  hubmatch 1.0

run_one chatmix_masked \
  outputs/block_qwen/ar2block_masked_1773298/checkpoints/last.ckpt \
  hubmatch 1.0

run_one mixedsft_masked \
  outputs/block_qwen/ar2block_masked_1836800/checkpoints/last.ckpt \
  hubmatch 1.0

run_one blocksizemix_masked \
  outputs/block_qwen/ar2block_masked_1855537/checkpoints/last.ckpt \
  hubmatch 1.0

run_one scratch_masked \
  outputs/block_qwen/block_masked_1836796/checkpoints/last.ckpt \
  hubmatch 1.0

# ---- Uniform / baseline ancestral (no thr) ----
run_one baseline_uniform \
  outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt \
  baseline

run_one shift_uniform \
  outputs/block_qwen/ar2block_uniform_1857278/checkpoints/last.ckpt \
  baseline

run_one anneal_uniform \
  outputs/block_qwen/ar2block_uniform_1857471/checkpoints/last.ckpt \
  baseline

run_one continueft_uniform \
  outputs/block_qwen/ar2block_uniform_1857519/checkpoints/last.ckpt \
  baseline

run_one joint_uniform \
  outputs/block_qwen/ar2block_uniform_1848844/checkpoints/last.ckpt \
  baseline

run_one blocksizemix_uniform \
  outputs/block_qwen/ar2block_uniform_1849287/checkpoints/last.ckpt \
  baseline

# ---- Hybrid (decode-as-masked / hubmatch) ----
run_one hybrid_anneal \
  outputs/block_qwen/ar2block_hybrid_1857475/checkpoints/last.ckpt \
  hubmatch 1.0

# ---- Compare panels (story pairs) ----
compare() {
  local left="$1" right="$2" out="$3"
  if [[ -d "${OUT_ROOT}/${left}" && -d "${OUT_ROOT}/${right}" ]]; then
    echo "=== compare ${left} vs ${right} ==="
    ./scripts/viz/make_decode_demo.sh --compare \
      "${OUT_ROOT}/${left}" "${OUT_ROOT}/${right}" "${OUT_ROOT}/${out}"
  else
    echo "SKIP compare ${out} (missing traces)"
  fi
}

compare baseline_masked baseline_uniform compare_baseline_masked_vs_baseline_uniform
compare baseline_masked shift_masked compare_baseline_vs_shift
compare baseline_masked anneal_masked compare_baseline_vs_anneal
compare baseline_masked joint_masked compare_baseline_vs_joint
compare baseline_masked mathmix_masked compare_baseline_vs_mathmix
compare baseline_masked complementary_masked compare_baseline_vs_comp
compare baseline_masked scratch_masked compare_baseline_vs_scratch
compare baseline_uniform shift_uniform compare_uniform_baseline_vs_shift
compare baseline_uniform anneal_uniform compare_uniform_baseline_vs_anneal
compare baseline_masked hybrid_anneal compare_baseline_vs_hybrid

echo "DONE"
ls -d "${OUT_ROOT}"/*/render/preview.gif 2>/dev/null | sort
ls -d "${OUT_ROOT}"/compare_*/preview.gif 2>/dev/null | sort
