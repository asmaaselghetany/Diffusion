#!/usr/bin/env bash
# Harvest parity / bake SUMMARYs into a quick TSV for AR2BLOCK_GSM_TABLE refresh.
set -euo pipefail
ROOTS=(
  /e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_masked_1762534
  /e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_uniform_1955203
  /e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_uniform_1855541
  /e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_masked_2012251
  /e/project1/scifi/elsayed3/Diffusion/outputs/block_qwen/ar2block_uniform_2012253
)
echo -e "dir\tgsm\tife"
for root in "${ROOTS[@]}"; do
  [[ -d "$root" ]] || continue
  while IFS= read -r -d '' md; do
    gsm=$(rg -o 'gsm8k\s*\|\s*[0-9.]+' "$md" | rg -o '[0-9.]+$' | tail -1 || true)
    ife=$(rg -o 'ifeval\s*\|\s*[0-9.]+' "$md" | rg -o '[0-9.]+$' | tail -1 || true)
    [[ -n "$gsm" ]] || continue
    echo -e "$(basename "$(dirname "$md")")\t${gsm}\t${ife:-}"
  done < <(find "$root" -name SUMMARY.md -print0 2>/dev/null)
done | sort
