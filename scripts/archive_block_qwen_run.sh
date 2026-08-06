#!/usr/bin/env bash
# After job 138144 (or any leftover run) finishes: move into uni-d2-data/micros and symlink back.
set -euo pipefail
DATA="${UNI_D2_DATA:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2-data}"
OUT="${UNI_D2_ROOT:-/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2}/outputs/block_qwen"
RUN="${1:?usage: $0 <run_dirname>}"

if [[ -L "$OUT/$RUN" ]]; then
  echo "already archived: $OUT/$RUN"; exit 0
fi
if [[ ! -d "$OUT/$RUN" ]]; then
  echo "missing: $OUT/$RUN"; exit 1
fi
# refuse if path looks actively written in last 2 minutes
if find "$OUT/$RUN" -type f -mmin -2 | grep -q .; then
  echo "refusing: files modified in last 2 minutes under $RUN"; exit 2
fi
# drop intermediate step ckpts
find "$OUT/$RUN/checkpoints" -name '0-*.ckpt' -delete 2>/dev/null || true
mkdir -p "$DATA/micros"
mv "$OUT/$RUN" "$DATA/micros/$RUN"
ln -s "$DATA/micros/$RUN" "$OUT/$RUN"
echo "archived → $DATA/micros/$RUN"
