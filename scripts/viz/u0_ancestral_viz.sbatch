#!/usr/bin/env bash
#SBATCH --job-name=u0_decode_viz
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --chdir=/e/project1/scifi/elsayed3/Diffusion-new
#
# Re-record U0 decode GIF with ancestral sampling (greedy soup archived).

set -euo pipefail
# shellcheck disable=SC1091
source scripts/_block_qwen_env.bash

CKPT="${CKPT:-outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt}"

# Preserve greedy-soup artifacts once
if [[ ! -d docs/research/decode_viz/baseline_uniform_greedy_soup ]]; then
  mkdir -p docs/research/decode_viz/baseline_uniform_greedy_soup
  cp -a docs/research/decode_viz/baseline_uniform/. \
    docs/research/decode_viz/baseline_uniform_greedy_soup/
fi

python scripts/viz/record_decode_trace.py \
  --ckpt "${CKPT}" \
  --label baseline_uniform \
  --out docs/research/decode_viz/baseline_uniform \
  --decode-profile baseline --no-greedy \
  --max-new-tokens 96 --num-steps 32 --device cuda --seed 0

python scripts/viz/render_decode_video.py \
  --trace docs/research/decode_viz/baseline_uniform \
  --out docs/research/decode_viz/baseline_uniform/render

python scripts/viz/compare_decode_traces.py \
  --left docs/research/decode_viz/baseline_masked \
  --right docs/research/decode_viz/baseline_uniform \
  --out docs/research/decode_viz/compare_baseline_masked_vs_baseline_uniform

python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('docs/research/decode_viz/baseline_uniform/meta.json').read_text())
print('greedy=', m.get('greedy'), 'profile=', m.get('decode_profile'))
print('final_text head:', (m.get('final_text') or '')[:240])
print('DONE')
PY
